"""
模型训练脚本。

流程：
1. 加载配置并固定随机种子
2. 准备标签映射产物
3. 加载预训练 BERT + 分类头
4. 分词、构建 DataLoader
5. 逐 epoch 训练（按 batch/step 写日志），验证集按 macro-F1 保存 best，并支持早停
6. 用 best 模型评估测试集，写出训练摘要与按 step 的曲线图

用法：
    python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from bert_tc.config import AppConfig, load_config
from bert_tc.utils.common import ensure_dir, save_csv, save_json, set_seed
from bert_tc.utils.metrics import compute_classification_metrics
from bert_tc.utils.viz import plot_confusion_matrix, plot_train_history
from prepare_data import prepare_dataset_artifacts


def get_device() -> torch.device:
    """优先使用 GPU，否则回退到 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer_and_model(
    config: AppConfig,
    metadata: dict,
    checkpoint_dir: Path | None = None,
):
    """
    加载 tokenizer 与序列分类模型。

    - checkpoint_dir 为 None：从预训练目录初始化（开始训练）
    - checkpoint_dir 有值：从已保存 checkpoint 恢复（评估）
    """
    model_source = checkpoint_dir if checkpoint_dir is not None else config.paths.pretrained_model_dir
    tokenizer = AutoTokenizer.from_pretrained(str(model_source))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_source),
        num_labels=metadata["num_labels"],
        id2label={int(key): value for key, value in metadata["id2label"].items()},
        label2id=metadata["label2id"],
    )
    return tokenizer, model


def tokenize_datasets(config: AppConfig, tokenizer):
    """
    对原始 DatasetDict 做批量分词，并整理为训练所需列。

    - 截断到 max_length
    - 标签列统一改名为 labels
    - 只保留模型输入相关列
    """
    raw_dataset = load_from_disk(str(config.paths.raw_data_dir))

    def tokenize_function(batch: dict) -> dict:
        return tokenizer(
            batch[config.data.text_column],
            truncation=True,
            max_length=config.data.max_length,
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        desc="对数据集进行分词编码",
    )
    if config.data.label_column != "labels":
        tokenized_dataset = tokenized_dataset.rename_column(config.data.label_column, "labels")

    keep_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized_dataset["train"].column_names:
        keep_columns.append("token_type_ids")

    remove_columns = [
        column_name
        for column_name in tokenized_dataset["train"].column_names
        if column_name not in keep_columns
    ]
    tokenized_dataset = tokenized_dataset.remove_columns(remove_columns)
    tokenized_dataset.set_format("python")
    return tokenized_dataset


def build_dataloaders(config: AppConfig, tokenized_dataset, tokenizer):
    """构建 train / validation / test 三个 DataLoader。"""
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    test_loader = DataLoader(
        tokenized_dataset["test"],
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    return train_loader, val_loader, test_loader


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """将 batch 中所有张量搬到指定设备。"""
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_model(model, data_loader, device: torch.device, label_names: list[str]) -> dict:
    """在给定 DataLoader 上评估，返回指标字典（含平均 loss）。"""
    model.eval()
    total_loss = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_labels.extend(batch["labels"].detach().cpu().tolist())

    metrics = compute_classification_metrics(
        labels=all_labels,
        predictions=all_predictions,
        label_names=label_names,
    )
    metrics["loss"] = total_loss / max(len(data_loader), 1)
    return metrics


def save_checkpoint(model, tokenizer, checkpoint_dir: Path, extra_metadata: dict) -> None:
    """以 HuggingFace 标准格式保存模型与分词器，并写入 metadata.json。"""
    ensure_dir(checkpoint_dir)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    save_json(extra_metadata, checkpoint_dir / "metadata.json")


def resolve_checkpoint_dir(config: AppConfig, checkpoint_name: str) -> Path:
    """例如 artifacts/checkpoints/best。"""
    return config.paths.checkpoints_dir / checkpoint_name


def evaluate_checkpoint(
    config_path: str = "configs/base.yaml",
    checkpoint_name: str = "best",
    split: str = "test",
) -> dict:
    """
    加载指定 checkpoint，在 validation 或 test 上评估并落盘。

    训练结束后也会调用本函数评估测试集；evaluate.py 直接复用。
    """
    config = load_config(config_path)
    metadata = prepare_dataset_artifacts(config)
    checkpoint_dir = resolve_checkpoint_dir(config, checkpoint_name)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_dir}")

    tokenizer, model = load_tokenizer_and_model(config, metadata, checkpoint_dir=checkpoint_dir)
    tokenized_dataset = tokenize_datasets(config, tokenizer)
    _, val_loader, test_loader = build_dataloaders(config, tokenized_dataset, tokenizer)

    device = get_device()
    model.to(device)

    if split == "validation":
        metrics = evaluate_model(model, val_loader, device, metadata["label_names"])
    elif split == "test":
        metrics = evaluate_model(model, test_loader, device, metadata["label_names"])
    else:
        raise ValueError("split 仅支持 validation 或 test")

    output_name = "test_metrics.json" if split == "test" else "validation_metrics.json"

    # 保存混淆矩阵热力图，便于直观查看分类对错分布
    cm_name = "test_confusion_matrix.png" if split == "test" else "validation_confusion_matrix.png"
    cm_path = plot_confusion_matrix(
        matrix=metrics["confusion_matrix"],
        label_names=metadata["label_names"],
        output_path=config.paths.reports_dir / cm_name,
        title=f"{split} Confusion Matrix",
    )
    metrics["confusion_matrix_plot"] = str(cm_path.resolve())
    save_json(metrics, config.paths.reports_dir / output_name)

    print(f"{split} 评估完成：accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")
    print(f"混淆矩阵图已保存：{cm_path}")
    return metrics


def train(config_path: str = "configs/base.yaml") -> dict:
    """端到端训练主流程。"""
    config = load_config(config_path)
    set_seed(config.seed)

    ensure_dir(config.paths.checkpoints_dir)
    ensure_dir(config.paths.reports_dir)
    ensure_dir(config.paths.logs_dir)

    # 1) 准备标签映射等元数据
    metadata = prepare_dataset_artifacts(config)

    # 2) 初始化模型与数据
    tokenizer, model = load_tokenizer_and_model(config, metadata)
    tokenized_dataset = tokenize_datasets(config, tokenizer)
    train_loader, val_loader, _ = build_dataloaders(config, tokenized_dataset, tokenizer)

    device = get_device()
    model.to(device)

    # 3) 优化器 + 线性 warmup 学习率调度
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_training_steps = len(train_loader) * config.training.num_epochs
    warmup_steps = int(total_training_steps * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    use_fp16 = config.training.use_fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # 按全局 step（batch）记录，便于 NLP 少 epoch 场景画细粒度曲线
    step_history_rows: list[dict] = []
    global_step = 0
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    # 4) 训练循环
    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        batch_in_epoch = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.num_epochs}")
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            predictions = outputs.logits.argmax(dim=-1)
            batch_size = batch["labels"].size(0)
            batch_correct = (predictions == batch["labels"]).sum().item()
            batch_acc = batch_correct / max(batch_size, 1)

            running_loss += loss.item()
            running_correct += batch_correct
            running_total += batch_size
            batch_in_epoch += 1
            global_step += 1

            # 每个 batch 记一行；验证指标仅在 epoch 结束时填到最后一行
            step_history_rows.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "batch": batch_in_epoch,
                    "train_loss": round(loss.item(), 6),
                    "train_accuracy": round(batch_acc, 6),
                    "val_loss": "",
                    "val_accuracy": "",
                    "val_f1_macro": "",
                }
            )

            progress_bar.set_postfix(
                loss=f"{running_loss / batch_in_epoch:.4f}",
                acc=f"{running_correct / max(running_total, 1):.4f}",
            )

        # 5) 验证 + 保存 checkpoint
        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = running_correct / max(running_total, 1)
        val_metrics = evaluate_model(model, val_loader, device, metadata["label_names"])

        # 把本轮验证结果挂到当前全局 step 对应的日志行上
        step_history_rows[-1]["val_loss"] = round(val_metrics["loss"], 6)
        step_history_rows[-1]["val_accuracy"] = round(val_metrics["accuracy"], 6)
        step_history_rows[-1]["val_f1_macro"] = round(val_metrics["f1_macro"], 6)
        save_csv(step_history_rows, config.paths.logs_dir / "train_history.csv")

        save_checkpoint(
            model,
            tokenizer,
            resolve_checkpoint_dir(config, "last"),
            {"saved_from_epoch": epoch, "type": "last"},
        )

        # 以验证集 macro-F1 选模并驱动早停
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model,
                tokenizer,
                resolve_checkpoint_dir(config, "best"),
                {"saved_from_epoch": epoch, "type": "best"},
            )
            save_json(val_metrics, config.paths.reports_dir / "best_val_metrics.json")
        else:
            patience_counter += 1

        print(
            f"[Epoch {epoch}] step={global_step} "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_f1={val_metrics['f1_macro']:.4f}"
        )

        if patience_counter >= config.training.patience:
            print(f"验证集指标连续 {config.training.patience} 轮未提升，提前停止训练。")
            break

    # 6) 绘制按 step 的训练曲线，并用 best 模型评估测试集
    history_csv = config.paths.logs_dir / "train_history.csv"
    curve_path = plot_train_history(
        history_csv_path=history_csv,
        output_path=config.paths.reports_dir / "train_curves.png",
    )
    if curve_path is not None:
        print(f"训练曲线图已保存：{curve_path}")

    test_metrics = evaluate_checkpoint(config_path, checkpoint_name="best", split="test")
    summary = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": round(best_val_f1, 6),
        "total_steps": global_step,
        "history_file": str(history_csv.resolve()),
        "train_curves_plot": str(curve_path.resolve()) if curve_path is not None else None,
        "best_checkpoint_dir": str(resolve_checkpoint_dir(config, "best").resolve()),
        "last_checkpoint_dir": str(resolve_checkpoint_dir(config, "last").resolve()),
        "test_metrics_file": str((config.paths.reports_dir / "test_metrics.json").resolve()),
        "test_confusion_matrix_plot": test_metrics.get("confusion_matrix_plot"),
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "test_f1_macro": round(test_metrics["f1_macro"], 6),
    }
    save_json(summary, config.paths.reports_dir / "training_summary.json")
    return summary


if __name__ == "__main__":
    summary = train()
    print("训练完成。")
    print(summary)

"""
评估与训练过程可视化。

生成：
- 混淆矩阵热力图
- 训练 / 验证曲线（默认按全局 step / batch 画，适配 NLP 少 epoch 场景）
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bert_tc.utils.common import ensure_dir


def _setup_chinese_font() -> None:
    """尽量使用系统中文字体，避免图里中文标签变成方框。"""
    # 按优先级尝试；找不到时 matplotlib 会回退到默认字体
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    # 否则负号可能显示成方块
    plt.rcParams["axes.unicode_minus"] = False


def _moving_average(values: list[float], window: int) -> list[float]:
    """简单滑动平均，用于压制单 batch 指标噪声。"""
    if window <= 1 or len(values) == 0:
        return values
    window = min(window, len(values))
    # 前缀和技巧：O(n) 算任意窗口均值
    cumsum = np.cumsum(np.insert(np.asarray(values, dtype=np.float64), 0, 0.0))
    # 前 window-1 个点用逐渐变长的窗口，避免开头出现空白
    smoothed: list[float] = []
    for index in range(1, len(values) + 1):
        left = max(0, index - window)
        smoothed.append(float((cumsum[index] - cumsum[left]) / (index - left)))
    return smoothed


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    label_names: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """
    将混淆矩阵画成热力图并保存为 PNG。

    行 = 真实标签，列 = 预测标签；格子上标注样本数。
    """
    _setup_chinese_font()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    cm = np.asarray(matrix, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(label_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # 深色格子用白字、浅色格子用黑字，保证数字可读
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)  # 及时释放，避免批量画图时内存涨
    return output_path


def plot_train_history(
    history_csv_path: str | Path,
    output_path: str | Path,
    smooth_window: int = 20,
) -> Path | None:
    """
    根据 train_history.csv 绘制训练曲线。

    新格式（推荐，按 step/batch）：
    - 左图：train_loss（原始 + 平滑）/ 验证 loss（epoch 末标记）
    - 右图：train_accuracy（平滑）/ val_accuracy / val_f1_macro

    旧格式（仅 epoch 汇总）仍可兼容绘制。
    """
    history_csv_path = Path(history_csv_path)
    if not history_csv_path.exists():
        return None

    with history_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None

    _setup_chinese_font()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # 有 step 列 => 新日志；否则走旧版 epoch 曲线
    has_step = "step" in rows[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if has_step:
        steps = [int(row["step"]) for row in rows]
        train_loss = [float(row["train_loss"]) for row in rows]
        train_acc = [float(row["train_accuracy"]) for row in rows]
        smooth_loss = _moving_average(train_loss, smooth_window)
        smooth_acc = _moving_average(train_acc, smooth_window)

        # 验证只在 epoch 末有值，其它行 val_* 为空字符串
        val_steps: list[int] = []
        val_loss: list[float] = []
        val_acc: list[float] = []
        val_f1: list[float] = []
        for row in rows:
            if row.get("val_loss") not in (None, ""):
                val_steps.append(int(row["step"]))
                val_loss.append(float(row["val_loss"]))
                val_acc.append(float(row["val_accuracy"]))
                val_f1.append(float(row["val_f1_macro"]))

        # 原始 batch loss 半透明，平滑曲线加粗，两者叠加更易读
        axes[0].plot(steps, train_loss, color="#93c5fd", linewidth=1.0, alpha=0.45, label="train_loss")
        axes[0].plot(steps, smooth_loss, color="#2563eb", linewidth=2.0, label=f"train_loss_ma{smooth_window}")
        if val_steps:
            axes[0].plot(val_steps, val_loss, color="#f97316", marker="o", linewidth=1.5, label="val_loss")
        axes[0].set_xlabel("Step (batch)")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curve")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].plot(steps, smooth_acc, color="#2563eb", linewidth=2.0, label=f"train_acc_ma{smooth_window}")
        if val_steps:
            axes[1].plot(val_steps, val_acc, color="#f97316", marker="o", linewidth=1.5, label="val_accuracy")
            axes[1].plot(val_steps, val_f1, color="#16a34a", marker="o", linewidth=1.5, label="val_f1_macro")
        axes[1].set_xlabel("Step (batch)")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Accuracy / F1 Curve")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.4)
    else:
        # 兼容旧版按 epoch 记录的 CSV（只有 3 个点那种）
        epochs = [int(row["epoch"]) for row in rows]
        train_loss = [float(row["train_loss"]) for row in rows]
        val_loss = [float(row["val_loss"]) for row in rows]
        train_acc = [float(row["train_accuracy"]) for row in rows]
        val_acc = [float(row["val_accuracy"]) for row in rows]
        val_f1 = [float(row["val_f1_macro"]) for row in rows]

        axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
        axes[0].plot(epochs, val_loss, marker="o", label="val_loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curve")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].plot(epochs, train_acc, marker="o", label="train_accuracy")
        axes[1].plot(epochs, val_acc, marker="o", label="val_accuracy")
        axes[1].plot(epochs, val_f1, marker="o", label="val_f1_macro")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Accuracy / F1 Curve")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

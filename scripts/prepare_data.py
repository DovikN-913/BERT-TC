"""
数据处理脚本。

读取已落盘的 HuggingFace DatasetDict，生成训练 / 推理都会用到的标签映射与统计信息：
- label2id.json
- id2label.json
- metadata.json（含各 split 长度统计与样例）

用法：
    python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Any

# 保证能 import 到 src/bert_tc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import load_from_disk

from bert_tc.config import AppConfig, load_config
from bert_tc.utils.common import ensure_dir, save_json


def get_label_names(dataset, label_column: str) -> list[str]:
    """
    解析标签名称列表。

    优先使用 datasets 特征中的 ClassLabel.names；
    若不存在（纯整型标签），则按训练集中出现过的标签排序后转成字符串。
    """
    label_feature = dataset["train"].features[label_column]
    # ClassLabel 会自带 names，例如 ["negative", "positive"]
    if hasattr(label_feature, "names"):
        return list(label_feature.names)

    # 纯 int 标签：排序后转字符串，保证顺序稳定（0,1,2...）
    unique_labels = sorted(set(dataset["train"][label_column]))
    return [str(label) for label in unique_labels]


def build_split_summary(split_dataset, text_column: str) -> dict[str, Any]:
    """
    统计单个 split 的样本量与文本长度分布，用于排查异常长度、指导 max_length。

    p95_text_length：约 95% 样本的文本长度不超过该值。
    """
    texts = split_dataset[text_column]
    # 这里用字符数近似文本长度（中文场景够用）；不是 tokenizer token 数
    lengths = [len(text.strip()) for text in texts]
    sorted_lengths = sorted(lengths)
    # 近似 95 分位下标；空集时至少落到 0，避免 IndexError
    p95_index = int(len(sorted_lengths) * 0.95) - 1
    p95_index = max(p95_index, 0)

    return {
        "num_rows": len(split_dataset),
        "avg_text_length": round(mean(lengths), 2),
        "max_text_length": max(lengths),
        "min_text_length": min(lengths),
        "p95_text_length": sorted_lengths[p95_index],
    }


def prepare_dataset_artifacts(config: AppConfig) -> dict:
    """
    读取原始数据集并写出处理产物，返回完整 metadata 字典。

    训练脚本也会调用本函数，确保标签映射与最新数据一致。
    """
    dataset = load_from_disk(str(config.paths.raw_data_dir))
    # 产物按数据集名分子目录，换数据集时不会互相覆盖
    output_dir = ensure_dir(config.paths.processed_dir / config.data.dataset_name)

    # 双向映射：训练用 label2id，推理展示用 id2label
    label_names = get_label_names(dataset, config.data.label_column)
    label2id = {label_name: index for index, label_name in enumerate(label_names)}
    # JSON 对象的 key 只能是字符串，所以 id 侧写成 "0"/"1"/...
    id2label = {str(index): label_name for label_name, index in label2id.items()}

    # 对各 split（train / validation / test）分别做长度统计
    split_summary = {}
    for split_name in dataset.keys():
        split_summary[split_name] = build_split_summary(
            dataset[split_name],
            config.data.text_column,
        )

    metadata = {
        "dataset_name": config.data.dataset_name,
        "text_column": config.data.text_column,
        "label_column": config.data.label_column,
        "num_labels": len(label_names),  # 二分类=2，多分类=N；训练时直接喂给模型
        "label_names": label_names,
        "label2id": label2id,
        "id2label": id2label,
        "splits": split_summary,
        # 附带一条训练集样例，方便人工快速确认「字段名 / 标签含义」是否对得上
        "sample": {
            "text": dataset["train"][0][config.data.text_column],
            "label": int(dataset["train"][0][config.data.label_column]),
            "label_name": label_names[int(dataset["train"][0][config.data.label_column])],
        },
    }

    save_json(label2id, output_dir / "label2id.json")
    save_json(id2label, output_dir / "id2label.json")
    save_json(metadata, output_dir / "metadata.json")
    return metadata


def main(config_path: str = "configs/base.yaml") -> None:
    """执行数据处理并打印摘要信息。"""
    config = load_config(config_path)
    metadata = prepare_dataset_artifacts(config)
    print("数据集处理完成。")
    print(f"数据集名称: {metadata['dataset_name']}")
    print(f"标签映射: {metadata['label2id']}")
    print(f"切分统计: {metadata['splits']}")


if __name__ == "__main__":
    main()

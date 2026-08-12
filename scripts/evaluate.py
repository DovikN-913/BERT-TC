"""
模型评估脚本。

加载指定 checkpoint，在 validation 或 test 上计算分类指标并写入 artifacts/reports/。
同时会生成混淆矩阵热力图：
- artifacts/reports/test_confusion_matrix.png
- artifacts/reports/validation_confusion_matrix.png

核心评估流程写在 train.py 的 evaluate_checkpoint 中（与训练结束后测测试集共用同一套逻辑），
本脚本负责命令行参数解析并调用它。

用法：
    python scripts/evaluate.py --checkpoint best --split test
    python scripts/evaluate.py --checkpoint best --split validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    """解析评估相关命令行参数。"""
    parser = argparse.ArgumentParser(description="评估指定 checkpoint 的分类效果")
    parser.add_argument("--checkpoint", default="best", help="checkpoint 名称，默认 best")
    parser.add_argument(
        "--split",
        default="test",
        choices=["validation", "test"],
        help="评估数据集切分",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate_checkpoint(checkpoint_name=args.checkpoint, split=args.split)
    print(metrics)

"""
单条文本预测脚本。

加载 artifacts/checkpoints 下的微调模型，对输入文本做分类，并输出标签与概率。

用法：
    python scripts/predict.py --text "这家酒店位置很好，服务也很周到。"
    python scripts/predict.py --text "..." --checkpoint best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bert_tc.config import AppConfig, load_config


class TextClassifierPredictor:
    """加载指定 checkpoint，对输入文本做情感（或其他）分类预测。"""

    def __init__(self, config: AppConfig, checkpoint_name: str | None = None) -> None:
        """
        初始化预测器。

        checkpoint_name 为空时，使用 config.serving.checkpoint_name（默认 best）。
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = checkpoint_name or config.serving.checkpoint_name
        self.checkpoint_dir = config.paths.checkpoints_dir / checkpoint
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"未找到可用模型目录: {self.checkpoint_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.checkpoint_dir))
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            int(label_id): label_name
            for label_id, label_name in self.model.config.id2label.items()
        }

    def predict(self, text: str) -> dict:
        """
        对单条文本执行预测。

        返回字段：
        - text / label / label_id / confidence / probabilities
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("输入文本不能为空。")

        encoded = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=self.config.data.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)
            predicted_id = int(torch.argmax(probabilities).item())

        probability_map = {
            self.id2label[index]: round(float(score), 6)
            for index, score in enumerate(probabilities.detach().cpu().tolist())
        }
        return {
            "text": clean_text,
            "label": self.id2label[predicted_id],
            "label_id": predicted_id,
            "confidence": round(float(probabilities[predicted_id].item()), 6),
            "probabilities": probability_map,
        }


def load_predictor(
    config_path: str = "configs/base.yaml",
    checkpoint_name: str | None = None,
) -> TextClassifierPredictor:
    """读配置并构造预测器（供本脚本 CLI 与 serve.py 共用）。"""
    config = load_config(config_path)
    return TextClassifierPredictor(config=config, checkpoint_name=checkpoint_name)


def parse_args() -> argparse.Namespace:
    """解析预测相关命令行参数。"""
    parser = argparse.ArgumentParser(description="对单条文本执行情感分类预测")
    parser.add_argument("--text", required=True, help="待预测文本")
    parser.add_argument("--checkpoint", default="best", help="checkpoint 名称，默认 best")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = load_predictor(checkpoint_name=args.checkpoint)
    result = predictor.predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

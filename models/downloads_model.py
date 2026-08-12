"""
从 ModelScope 下载中文 BERT 预训练模型到本地。

用法（在项目根目录执行）：
    python models/downloads_model.py

说明：
- 依赖 modelscope 包（不在 requirements.txt 主流程里，仅下载时需要）；
- cache_dir="./" 表示把模型缓存到当前工作目录；
- 下载完成后，configs/base.yaml 中的 pretrained_model_dir
  通常指向 models/models/google-bert--bert-base-chinese。
"""

from modelscope import snapshot_download

# 下载 google-bert/bert-base-chinese；返回本地模型目录路径
model_dir = snapshot_download("google-bert/bert-base-chinese", cache_dir="./")
print(f"模型已下载到: {model_dir}")

"""
通用工具函数。

涵盖：
- 目录创建
- JSON / CSV 读写
- 训练可复现所需的随机种子设置
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在（含多级父目录），并返回 Path 对象。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict, path: str | Path) -> None:
    """
    将字典保存为 UTF-8 JSON 文件。

    - ensure_ascii=False：保留中文可读性
    - indent=2：便于人工查看
    """
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    """从 UTF-8 JSON 文件读取并返回 dict。"""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_csv(rows: list[dict], path: str | Path) -> None:
    """
    将「字典列表」写为 CSV。

    约定：
    - 表头取第一行字典的全部键；
    - 使用 utf-8-sig，方便 Excel 直接打开中文不乱码；
    - rows 为空时直接返回，不创建空文件内容。
    """
    output_path = Path(path)
    ensure_dir(output_path.parent)
    if not rows:
        return

    # newline=""：避免 Windows 下 DictWriter 多空行
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    """
    设置 Python / NumPy / PyTorch（含 CUDA）随机种子，提升实验可复现性。

    注意：完全可复现还受 cuDNN、多进程 DataLoader 等因素影响，此处覆盖最常见来源。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""
配置加载模块。

职责：
1. 用 Pydantic 定义各配置段的数据结构与默认值；
2. 从 YAML 文件读取配置；
3. 将相对路径统一解析为基于项目根目录的绝对路径，避免在不同工作目录下运行脚本时路径失效。
"""

from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel, Field


# config.py 位于 src/bert_tc/ 下，向上两级即项目根目录 BERT-TC/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 泛型约束：仅允许传入继承自 BaseModel 的配置类
ModelT = TypeVar("ModelT", bound=BaseModel)


class PathsConfig(BaseModel):
    """项目中各类路径配置（相对路径会在 load_config 中解析为绝对路径）。"""

    raw_data_dir: Path          # 原始数据集目录（如 data/ChnSentiCorp）
    pretrained_model_dir: Path  # 本地预训练 BERT 模型目录
    processed_dir: Path         # 数据处理产物目录（label 映射、metadata 等）
    checkpoints_dir: Path       # 训练保存的 checkpoint 根目录
    reports_dir: Path           # 评估报告 / 训练摘要输出目录
    logs_dir: Path              # 训练日志（如 train_history.csv）输出目录


class DataConfig(BaseModel):
    """与数据集字段、分词长度相关的配置。"""

    dataset_name: str                 # 数据集名称，用于产物子目录命名
    text_column: str = "text"         # 文本字段名
    label_column: str = "label"       # 标签字段名
    max_length: int = 256             # tokenizer 截断最大长度


class TrainingConfig(BaseModel):
    """训练超参数与工程相关设置。"""

    batch_size: int = 16              # 训练 batch 大小
    eval_batch_size: int = 32         # 验证 / 测试 batch 大小（可不等于训练 batch）
    learning_rate: float = 2.0e-5     # AdamW 学习率（BERT 微调常用量级）
    weight_decay: float = 0.01        # 权重衰减，缓解过拟合
    num_epochs: int = 3               # 最大训练轮数（可能被早停提前终止）
    warmup_ratio: float = 0.1         # 学习率预热步数占总训练步数的比例
    gradient_clip_norm: float = 1.0   # 梯度裁剪阈值，防止梯度爆炸
    patience: int = 2                 # 早停耐心值：验证指标连续多少轮不提升则停止
    num_workers: int = 0              # DataLoader 工作进程数；Windows 下常用 0
    use_fp16: bool = False            # 是否启用 CUDA 混合精度训练


class ServingConfig(BaseModel):
    """推理服务启动与默认加载 checkpoint 的配置。"""

    host: str = "127.0.0.1"           # 服务监听地址
    port: int = 8000                  # 服务监听端口
    checkpoint_name: str = "best"     # 默认加载的 checkpoint 子目录名


class AppConfig(BaseModel):
    """整份 YAML 对应的顶层配置对象。"""

    project_name: str = "bert-tc"
    seed: int = 42                    # 全局随机种子，保证实验可复现
    paths: PathsConfig
    data: DataConfig
    # 允许 YAML 中省略 training / serving 段，此时使用类内默认值
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)


def _validate_model(model_cls: Type[ModelT], data: dict) -> ModelT:
    """
    兼容不同 pydantic 大版本的校验入口。

    - pydantic v2：使用 model_validate
    - pydantic v1：使用 parse_obj
    """
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def _resolve_path(path_value: Path) -> Path:
    """将相对路径转为基于 PROJECT_ROOT 的绝对路径；绝对路径原样返回。"""
    if path_value.is_absolute():
        return path_value
    return (PROJECT_ROOT / path_value).resolve()


def load_config(config_path: str | Path = "configs/base.yaml") -> AppConfig:
    """
    加载 YAML 配置并返回 AppConfig。

    步骤：
    1. 解析配置文件路径（相对路径基于项目根目录）；
    2. 读取 YAML 为 dict；
    3. 用 Pydantic 校验并构造 AppConfig；
    4. 将 paths 下所有路径统一解析为绝对路径后返回。
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (PROJECT_ROOT / config_file).resolve()

    with config_file.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    config = _validate_model(AppConfig, raw_config)
    config.paths.raw_data_dir = _resolve_path(config.paths.raw_data_dir)
    config.paths.pretrained_model_dir = _resolve_path(config.paths.pretrained_model_dir)
    config.paths.processed_dir = _resolve_path(config.paths.processed_dir)
    config.paths.checkpoints_dir = _resolve_path(config.paths.checkpoints_dir)
    config.paths.reports_dir = _resolve_path(config.paths.reports_dir)
    config.paths.logs_dir = _resolve_path(config.paths.logs_dir)
    return config

# BERT-TC

一个面向中文文本分类微调任务的学习向项目示例。当前版本围绕 `ChnSentiCorp` 情感分类数据集，提供从数据集处理、模型训练、模型评估、单条预测，到 FastAPI + 网页测试页的完整闭环。

业务主流程直接写在 `scripts/` 里，打开脚本即可顺着读完；`src/bert_tc/` 只放配置与通用工具，避免样板代码重复。

## 1. 项目目标

1. 数据集准备与统计
2. BERT 分类模型微调
3. 验证集 / 测试集评估
4. 保存最佳模型与训练日志
5. 启动本地 API 服务
6. 通过网页输入文本完成在线测试

## 2. 项目结构

```text
BERT-TC/
├─ configs/
│  └─ base.yaml                 # 统一配置文件
├─ scripts/
│  ├─ prepare_data.py           # 数据处理（完整逻辑）
│  ├─ train.py                  # 训练 + 评估辅助（完整逻辑）
│  ├─ evaluate.py               # 评估入口（复用 train.py）
│  ├─ predict.py                # 单条预测（含 Predictor）
│  └─ serve.py                  # FastAPI 服务（完整路由）
├─ src/
│  └─ bert_tc/
│     ├─ config.py              # 配置加载
│     └─ utils/
│        ├─ common.py           # 通用工具
│        ├─ metrics.py          # 分类指标
│        └─ viz.py              # 混淆矩阵 / 训练曲线可视化
├─ web/
│  └─ static/
│     ├─ index.html             # 网页测试页
│     ├─ styles.css             # 页面样式
│     └─ app.js                 # 页面逻辑
├─ artifacts/
│  ├─ processed/                # 数据处理产物
│  ├─ checkpoints/              # 模型权重
│  ├─ reports/                  # 评估报告
│  └─ logs/                     # 训练日志
├─ data/ChnSentiCorp/           # 原始数据集
├─ models/                      # 本地预训练模型
├─ docs/
│  ├─ execution_flow.md          # 执行流程（从零跑通）
│  └─ finetune_other_datasets.md # 换数据集 / 多分类微调指南
└─ requirements.txt
```

## 3. 环境准备

建议使用你当前的 `bert-tc` conda 环境：

```powershell
conda activate bert-tc
pip install -r requirements.txt
```

## 4. 配置说明

默认配置文件是 `configs/base.yaml`，主要包含：

- 数据集路径
- 预训练模型路径
- 最大文本长度
- batch size
- 学习率
- epoch 数
- 提前停止 patience
- 服务启动地址与端口

## 5. 运行流程

### 5.1 数据处理

```powershell
python scripts/prepare_data.py
```

运行后会在 `artifacts/processed/ChnSentiCorp/` 下生成：

- `label2id.json`
- `id2label.json`
- `metadata.json`

### 5.2 训练模型

```powershell
python scripts/train.py
```

训练完成后会生成：

- `artifacts/checkpoints/best/`
- `artifacts/checkpoints/last/`
- `artifacts/logs/train_history.csv`
- `artifacts/reports/best_val_metrics.json`
- `artifacts/reports/test_metrics.json`
- `artifacts/reports/training_summary.json`
- `artifacts/reports/train_curves.png`
- `artifacts/reports/test_confusion_matrix.png`

### 5.3 评估模型

评估最佳模型在测试集上的效果：

```powershell
python scripts/evaluate.py --checkpoint best --split test
```

评估验证集：

```powershell
python scripts/evaluate.py --checkpoint best --split validation
```

### 5.4 单条文本预测

```powershell
python scripts/predict.py --text "这家酒店位置很好，服务也很周到。"
```

### 5.5 启动网页服务

```powershell
python scripts/serve.py
```

启动后访问：

```text
http://127.0.0.1:8000
```

即可在网页上输入文本完成预测测试。

## 6. 产物说明

### 模型产物

- `artifacts/checkpoints/best/`：验证集上表现最好的模型
- `artifacts/checkpoints/last/`：训练结束时最后一次保存的模型

### 日志与报告

- `train_history.csv`：按全局 step（batch）记录的训练指标；验证指标写在每个 epoch 末对应 step 上
- `best_val_metrics.json`：最佳验证集指标
- `test_metrics.json`：测试集评估指标
- `training_summary.json`：训练摘要
- `train_curves.png`：以 Step(batch) 为横轴的训练 / 验证曲线
- `test_confusion_matrix.png` / `validation_confusion_matrix.png`：混淆矩阵热力图

## 7. 指标说明

项目默认输出以下分类指标：

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion Matrix
- Classification Report

## 8. 说明

优先阅读 `scripts/` 下的脚本理解完整流程；`src/bert_tc/` 仅提供配置与工具函数。

- 执行流程（命令顺序与产物）：[docs/execution_flow.md](docs/execution_flow.md)
- 换数据集或做多分类微调：[docs/finetune_other_datasets.md](docs/finetune_other_datasets.md)

# 其他数据集上的 BERT 微调指南

本文说明：如何把本项目从当前的 `ChnSentiCorp`（二分类情感）切换到**其他数据集**，包括**多分类**任务。

本仓库已经按「文本分类」通用流程写好了训练代码：

- 标签数量 `num_labels` 会从数据里自动统计
- 分类头、评估指标（含 macro-F1）、混淆矩阵都会跟着类别数变化
- **多数情况下只改数据 + 配置，不必改模型代码**

---

## 1. 项目对数据的约定

训练脚本通过 `datasets.load_from_disk` 读取数据，因此目标格式是 HuggingFace `DatasetDict` 落盘目录，且至少包含：

| split | 是否必需 | 用途 |
|-------|----------|------|
| `train` | 必需 | 训练 |
| `validation` | 必需 | 验证、早停、保存 best |
| `test` | 必需 | 最终测试评估 |

每条样本至少包含：

| 字段 | 含义 | 配置项 |
|------|------|--------|
| 文本列 | 待分类文本（字符串） | `data.text_column`，默认 `text` |
| 标签列 | 类别 id（整数）或 `ClassLabel` | `data.label_column`，默认 `label` |

目录示例：

```text
data/YourDataset/
├─ dataset_dict.json
├─ train/
├─ validation/
└─ test/
```

> 若你只有训练集，需要自己切分出 `validation` / `test`，否则当前脚本无法直接跑通。

---

## 2. 二分类 vs 多分类：你要改什么？

| 项目 | 二分类（当前） | 多分类（N 类，N≥3） |
|------|----------------|---------------------|
| 标签取值 | `0/1` | `0..N-1` |
| `num_labels` | 自动为 2 | 自动为 N |
| 模型代码 | 不用改 | 不用改 |
| 损失函数 | CrossEntropy（transformers 内置） | 同左，自动适配 |
| 评估指标 | accuracy / macro-P/R/F1 | 同左；类别不均衡时更看重 **macro-F1** |
| 配置 | 改路径与字段名 | 同左，并建议检查 `max_length`、`batch_size` |

结论：**多分类不是另写一套训练逻辑，关键是标签从 0 连续编号，并让配置指向新数据。**

---

## 3. 操作总流程（推荐顺序）

```text
准备数据落盘
  → 修改 configs/base.yaml
  → python scripts/prepare_data.py   # 确认标签与长度统计
  → 清理旧 checkpoint（重要）
  → python scripts/train.py
  → python scripts/evaluate.py
  → python scripts/predict.py / serve.py
```

---

## 4. 准备新数据集

### 4.1 已有 HuggingFace 数据集（推荐）

例如从 Hub 拉取后保存到本地：

```python
from datasets import load_dataset

# 示例：换成你的数据集名
ds = load_dataset("your_org/your_dataset")

# 若只有 train，可自行切分
if "validation" not in ds or "test" not in ds:
    split = ds["train"].train_test_split(test_size=0.2, seed=42)
    temp = split["test"].train_test_split(test_size=0.5, seed=42)
    from datasets import DatasetDict
    ds = DatasetDict(
        {
            "train": split["train"],
            "validation": temp["train"],
            "test": temp["test"],
        }
    )

# 若字段名不是 text / label，可先 rename
# ds = ds.rename_columns({"content": "text", "category": "label"})

ds.save_to_disk("data/YourDataset")
```

### 4.2 从 CSV / JSON / Excel 构建

最小示例（CSV 含 `text,label`）：

```python
from datasets import load_dataset, ClassLabel, DatasetDict

# label 建议已是 0..N-1 的整数
raw = load_dataset("csv", data_files={
    "train": "raw/train.csv",
    "validation": "raw/val.csv",
    "test": "raw/test.csv",
})

# 可选：把整数标签变成带名字的 ClassLabel，便于报告可读
label_names = ["体育", "财经", "科技", "娱乐"]  # 按你的类别改
class_label = ClassLabel(names=label_names)

def cast_label(example):
    return example

raw = raw.cast_column("label", class_label)
raw.save_to_disk("data/YourDataset")
```

### 4.3 标签编码注意点

1. **标签必须是从 0 开始的连续整数**（`0,1,2,...,N-1`）。  
   不要用 `1..N`，也不要出现跳号（如 `0,1,3`）。
2. 字符串类别要先映射成整数，例如：

```python
label2id = {"体育": 0, "财经": 1, "科技": 2, "娱乐": 3}

def encode(example):
    example["label"] = label2id[example["label"]]
    return example

ds = ds.map(encode)
```

3. 本项目的 `prepare_data.py` 会：
   - 优先读取 `ClassLabel.names`
   - 否则用训练集中出现过的标签排序后生成名称

---

## 5. 修改配置文件

编辑 [`configs/base.yaml`](../configs/base.yaml)，至少改这些：

```yaml
paths:
  raw_data_dir: data/YourDataset          # 新数据集目录
  pretrained_model_dir: models/models/google-bert--bert-base-chinese

data:
  dataset_name: YourDataset               # 产物子目录名
  text_column: text                       # 你的文本列名
  label_column: label                     # 你的标签列名
  max_length: 256                         # 参考 prepare 后的 p95 长度再调

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 3
  patience: 2
```

### 5.1 中文 / 英文数据集怎么选预训练模型

| 语种 | 建议模型 |
|------|----------|
| 中文 | `bert-base-chinese`（本项目默认） |
| 英文 | `bert-base-uncased` / `bert-base-cased` |
| 中英混合 | 可考虑 `bert-base-multilingual-cased` |

换模型后，把下载好的本地路径填到 `paths.pretrained_model_dir`。

下载中文 BERT 仍可用：

```powershell
python models/downloads_model.py
```

---

## 6. 检查数据是否就绪

```powershell
python scripts/prepare_data.py
```

重点看终端输出和：

`artifacts/processed/YourDataset/metadata.json`

确认：

- `num_labels` 是否等于你的类别数
- `label_names` / `label2id` 是否正确
- 各 split 的 `num_rows`、`p95_text_length` 是否合理

若 `p95_text_length` 远大于当前 `max_length`，可适当增大 `max_length`（显存不够就减小 `batch_size`）。

---

## 7. 清理旧产物后再训练（很重要）

换数据集或改类别数后，**不要复用旧 checkpoint**：

- 旧模型分类头维度可能是 2，新任务是 N 类，会直接报错或结果错乱
- 旧的 `label2id` 可能与新数据不一致

建议清理：

```powershell
Remove-Item -Recurse -Force artifacts\checkpoints\*
Remove-Item -Recurse -Force artifacts\reports\*
Remove-Item -Recurse -Force artifacts\logs\*
```

然后训练：

```powershell
python scripts/train.py
```

评估 / 预测 / 服务：

```powershell
python scripts/evaluate.py --checkpoint best --split test
python scripts/predict.py --text "这里填一条新领域文本"
python scripts/serve.py
```

---

## 8. 多分类场景的调参建议

1. **类别不均衡**  
   - 继续看 `macro-F1`（本项目早停已按它选模）  
   - 极度不均衡时可考虑过采样、加权采样或 class weight（当前代码未内置，需自行扩展）

2. **类别很多（如 20+）**  
   - 适当增加 `num_epochs` 或减小 `learning_rate`（如 `1e-5`）  
   - 关注混淆矩阵，看哪些类容易互相误判

3. **文本很长**  
   - `max_length` 提到 `384/512`  
   - 同步减小 `batch_size`，必要时开启 `use_fp16: true`（需 CUDA）

4. **英文短文本**  
   - `max_length` 可用 `64/128`，训练更快

---

## 9. 常见问题

### Q1：报错找不到 `validation` / `test`

当前训练脚本固定读取这三个 split。请先按第 4 节补齐切分。

### Q2：`num_labels` 不对

检查：

- 标签是否从 0 连续编号
- `label_column` 是否指对列
- 是否误把字符串标签直接训练（应先映射为整数或 `ClassLabel`）

### Q3：换数据后预测标签还是 negative/positive

通常是还在加载旧的 `artifacts/checkpoints/best`。清掉旧 checkpoint 后重新训练。

### Q4：字段不叫 `text` / `label`

两种办法：

1. 改配置里的 `text_column` / `label_column`
2. 或在保存数据前 `rename_columns`

### Q5：想同时保留多套实验

可为不同数据集准备不同配置文件，例如 `configs/tnews.yaml`，训练时在代码入口传入该路径；或复制一份 `artifacts` 目录做实验隔离。当前默认入口读取 `configs/base.yaml`。

---

## 10. 最小检查清单

- [ ] 数据已是 `train/validation/test` 的 `save_to_disk` 目录
- [ ] 文本列、标签列名称与配置一致
- [ ] 标签为 `0..N-1`
- [ ] `prepare_data.py` 打印的 `num_labels` 正确
- [ ] 已清理旧 checkpoint
- [ ] `pretrained_model_dir` 与语种匹配
- [ ] 训练后查看 `train_curves.png` 与混淆矩阵

完成以上步骤后，即可在本项目中对任意文本多分类数据集做 BERT 微调。

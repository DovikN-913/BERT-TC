"""
分类任务评估指标计算。

基于 sklearn，输出训练 / 评估流程中统一使用的指标字典，
便于写入 JSON 报告与控制台打印。
"""

from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    label_names: list[str],
) -> dict:
    """
    根据真实标签与预测标签计算分类指标。

    返回字段说明：
    - accuracy：整体准确率
    - precision_macro / recall_macro / f1_macro：宏平均（各类别平等加权）
    - confusion_matrix：混淆矩阵（二维 list）
    - classification_report：按类别拆开的详细报告（dict 形式）

    zero_division=0：某类无预测样本时，相关比率记为 0，避免告警中断流程。
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",  # 各类别等权平均；类别不均衡时比 micro / 整体 accuracy 更公平
        zero_division=0,
    )
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        output_dict=True,  # 返回 dict，方便写 JSON
        zero_division=0,
    )
    # tolist() 便于后续 json.dump 序列化（numpy 数组不能直接 dump）
    matrix = confusion_matrix(labels, predictions).tolist()

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": matrix,
        "classification_report": report,
    }

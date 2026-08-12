/**
 * 网页演示页前端逻辑。
 * 负责：读取输入 -> 调用 /predict -> 渲染标签、置信度与各类别概率。
 */

// DOM 元素引用
const textInput = document.getElementById("text-input");
const predictButton = document.getElementById("predict-btn");
const statusBox = document.getElementById("status");
const resultCard = document.getElementById("result-card");
const labelSpan = document.getElementById("label");
const confidenceSpan = document.getElementById("confidence");
const probabilityList = document.getElementById("probability-list");

/**
 * 发起一次预测请求并更新页面。
 */
async function predictText() {
  const text = textInput.value.trim();
  if (!text) {
    statusBox.textContent = "请输入要预测的文本。";
    resultCard.classList.add("hidden");
    return;
  }

  // 请求进行中禁用按钮，避免重复提交
  predictButton.disabled = true;
  statusBox.textContent = "模型正在预测，请稍候...";
  resultCard.classList.add("hidden");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    if (!response.ok) {
      // FastAPI 错误体通常带 detail 字段
      throw new Error(data.detail || "预测失败");
    }

    // 渲染主结果
    labelSpan.textContent = data.label;
    confidenceSpan.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    probabilityList.innerHTML = "";

    // 渲染每个类别的概率列表
    Object.entries(data.probabilities).forEach(([label, score]) => {
      const item = document.createElement("li");
      item.textContent = `${label}: ${(score * 100).toFixed(2)}%`;
      probabilityList.appendChild(item);
    });

    statusBox.textContent = "预测完成。";
    resultCard.classList.remove("hidden");
  } catch (error) {
    statusBox.textContent = error.message;
    resultCard.classList.add("hidden");
  } finally {
    predictButton.disabled = false;
  }
}

predictButton.addEventListener("click", predictText);

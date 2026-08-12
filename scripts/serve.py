"""
本地推理服务启动脚本。

提供：
- GET  /          ：网页演示首页
- GET  /health    ：健康检查
- POST /predict   ：文本分类推理
- /static/*       ：前端静态资源

用法：
    python scripts/serve.py

启动后默认访问：http://127.0.0.1:8000
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bert_tc.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT
from bert_tc.config import load_config
from predict import load_predictor


class PredictRequest(BaseModel):
    """预测请求体。"""

    text: str = Field(..., description="待分类的中文文本")


class PredictResponse(BaseModel):
    """预测响应体，字段与 TextClassifierPredictor.predict 返回值对齐。"""

    text: str
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]


def create_app(config_path: str = "configs/base.yaml") -> FastAPI:
    """
    创建 FastAPI 应用。

    模型在 startup 时加载一次，后续请求复用同一 predictor。
    """
    config = load_config(config_path)
    app = FastAPI(title="BERT 中文分类服务", version="1.0.0")

    static_dir = CONFIG_PROJECT_ROOT / "web" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("startup")
    def load_runtime_objects() -> None:
        """服务启动时预加载模型。"""
        app.state.predictor = load_predictor(config_path=config_path)

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        """返回网页测试页。"""
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/health")
    def health() -> dict:
        """健康检查。"""
        return {"status": "ok", "project": config.project_name}

    @app.post("/predict", response_model=PredictResponse)
    def predict_api(request: PredictRequest) -> PredictResponse:
        """
        文本分类接口。

        - 空文本等业务错误 -> 400
        - checkpoint 缺失等服务端问题 -> 500
        """
        try:
            result = app.state.predictor.predict(request.text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PredictResponse(**result)

    return app


if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        create_app(),
        host=config.serving.host,
        port=config.serving.port,
    )

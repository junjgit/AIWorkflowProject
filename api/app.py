"""Flask API for training, prediction, logs, and monitoring metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, request

from src.logging_utils import read_log
from src.modeling import load_manifest, predict_revenue, train_all_models

ROOT = Path(__file__).resolve().parents[1]


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        TRAIN_DATA_DIR=str(ROOT / "cs-train"),
        PRODUCTION_DATA_DIR=str(ROOT / "cs-production"),
        MODEL_DIR=str(ROOT / "models"),
        LOG_DIR=str(ROOT / "logs"),
        METRICS_DIR=str(ROOT / "reports" / "metrics"),
        MODEL_COMPARISON=str(ROOT / "reports" / "metrics" / "model_comparison.csv"),
    )
    if config:
        app.config.update(config)

    @app.get("/health")
    def health():
        try:
            manifest = load_manifest(app.config["MODEL_DIR"])
            model_count = len(manifest["countries"])
            version = manifest["model_version"]
        except FileNotFoundError:
            model_count = 0
            version = None
        return jsonify({"status": "ok", "models_available": model_count, "model_version": version})

    @app.post("/train")
    def train():
        result = train_all_models(
            app.config["TRAIN_DATA_DIR"],
            app.config["MODEL_DIR"],
            app.config["LOG_DIR"],
            app.config["MODEL_COMPARISON"],
        )
        return jsonify(result), 200

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        if "country" not in payload or "date" not in payload:
            return jsonify({"error": "Request JSON must include 'country' and 'date'."}), 400
        try:
            result = predict_revenue(
                country=payload["country"],
                forecast_date=payload["date"],
                data_dirs=[app.config["TRAIN_DATA_DIR"], app.config["PRODUCTION_DATA_DIR"]],
                model_dir=app.config["MODEL_DIR"],
                log_dir=app.config["LOG_DIR"],
            )
            return jsonify(result), 200
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/logfile")
    def logfile():
        log_type = request.args.get("type", "predict")
        try:
            records = read_log(app.config["LOG_DIR"], log_type)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"log_type": log_type, "records": records, "count": len(records)})

    @app.get("/logs/train")
    def train_log():
        records = read_log(app.config["LOG_DIR"], "train")
        return jsonify({"records": records, "count": len(records)})

    @app.get("/logs/predict")
    def predict_log():
        records = read_log(app.config["LOG_DIR"], "predict")
        return jsonify({"records": records, "count": len(records)})

    @app.get("/metrics")
    def metrics():
        path = Path(app.config["METRICS_DIR"]) / "post_production_summary.json"
        if not path.exists():
            return jsonify({"error": "Monitoring metrics not found. Run scripts/post_production.py."}), 404
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Endpoint not found"}), 404

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

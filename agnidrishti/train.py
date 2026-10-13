"""Train, evaluate and explain models. CLI: ``python -m agnidrishti.train``.

Outputs: models/xgb_v1.joblib + .json, reports/metrics.json + PNGs,
data/processed/predictions.parquet and shap_top5.parquet.
"""

import itertools
import json
import logging
from datetime import UTC, datetime

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from agnidrishti import config
from agnidrishti.evaluate import compute_metrics, plot_calibration, plot_pr
from agnidrishti.explain import top_k_contributions
from agnidrishti.features import DATASET_PATH, FEATURES
from agnidrishti.utils import setup_logging

log = logging.getLogger(__name__)

PREDICTIONS_PATH = config.DATA_PROCESSED / "predictions.parquet"
SHAP_PATH = config.DATA_PROCESSED / "shap_top5.parquet"
PARAM_GRID = {
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
    "min_child_weight": [1, 5],
}


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal train/val/test split on the year of ``week_start``; never shuffles."""
    year = df["week_start"].dt.year

    def between(span: tuple[int, int]) -> pd.DataFrame:
        return df[(year >= span[0]) & (year <= span[1])].sort_values(["week_start", "cell_id"])

    return between(config.TRAIN_YEARS), between(config.VAL_YEARS), between(config.TEST_YEARS)


def train_logreg(train: pd.DataFrame) -> object:
    """Balanced logistic-regression baseline with median imputation and scaling."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )
    return model.fit(train[FEATURES], train["y"])


def tune_xgb(train: pd.DataFrame, val: pd.DataFrame) -> tuple[XGBClassifier, dict, float]:
    """Grid-search XGBoost on validation PR-AUC; returns (model, params, val_pr_auc)."""
    spw = float((train["y"] == 0).sum() / max((train["y"] == 1).sum(), 1))
    best: tuple[XGBClassifier | None, dict, float] = (None, {}, -1.0)
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID, values, strict=True))
        params |= {"scale_pos_weight": round(spw, 3), "subsample": 0.8,
                   "colsample_bytree": 0.8, "eval_metric": "aucpr", "random_state": 42}
        model = XGBClassifier(**params, n_jobs=-1)
        model.fit(train[FEATURES], train["y"])
        score = average_precision_score(val["y"], model.predict_proba(val[FEATURES])[:, 1])
        log.info("XGB %s -> val PR-AUC %.4f", values, score)
        if score > best[2]:
            best = (model, params, score)
    return best  # type: ignore[return-value]


def main() -> None:
    """CLI entry point: run the full training, evaluation and explanation pipeline."""
    setup_logging()
    for d in (config.MODELS_DIR, config.REPORTS_DIR):
        d.mkdir(exist_ok=True)
    df = pd.read_parquet(DATASET_PATH)
    train, val, test = split_by_time(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info("%s: %d rows, %d positives, %s..%s", name, len(part), part["y"].sum(),
                 part["week_start"].min().date(), part["week_start"].max().date())

    logreg = train_logreg(train)
    xgb, params, val_pr = tune_xgb(train, val)
    log.info("Best XGB params %s (val PR-AUC %.4f)", params, val_pr)

    metrics: dict = {}
    curves = {}
    for name, model in (("logreg", logreg), ("xgboost", xgb)):
        metrics[name] = {}
        for split, part in (("val", val), ("test", test)):
            p = model.predict_proba(part[FEATURES])[:, 1]
            metrics[name][split] = compute_metrics(part["y"].to_numpy(), p)
            if split == "test":
                curves[name] = (part["y"].to_numpy(), p)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_calibration(curves, str(config.REPORTS_DIR / "calibration_test.png"))
    plot_pr(curves, str(config.REPORTS_DIR / "pr_curve_test.png"))
    log.info("Metrics:\n%s", json.dumps(metrics, indent=2))

    model_path = config.MODELS_DIR / f"{config.MODEL_NAME}.joblib"
    joblib.dump(xgb, model_path)
    joblib.dump(logreg, config.MODELS_DIR / "logreg_baseline.joblib")
    sidecar = {
        "version": config.MODEL_NAME,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURES,
        "training_window": {"train": config.TRAIN_YEARS, "val": config.VAL_YEARS,
                            "test": config.TEST_YEARS},
        "hyperparameters": params,
        "metrics": metrics,
    }
    model_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    log.info("Saved model to %s", model_path)

    served = pd.concat([val, test])
    preds = served[["cell_id", "week_start", "y"]].copy()
    preds["prob"] = xgb.predict_proba(served[FEATURES])[:, 1]
    preds.to_parquet(PREDICTIONS_PATH, index=False)
    top5 = top_k_contributions(xgb, served[FEATURES], served[["cell_id", "week_start"]])
    top5.to_parquet(SHAP_PATH, index=False)
    log.info("Saved %d predictions and SHAP top-5 for %d dates",
             len(preds), preds["week_start"].nunique())


if __name__ == "__main__":
    main()

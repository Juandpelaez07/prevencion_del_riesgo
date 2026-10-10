"""Entrenamiento y comparación de los tres clasificadores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.schema import CATEGORICAL, FEATURES, LABELS, NUMERIC, TARGET

MODEL_ORDER = ["Regresión logística", "Random Forest", "XGBoost"]


class ModelingError(ValueError):
    pass


@dataclass
class ModelReport:
    name: str
    pipeline: Pipeline
    threshold: float
    metrics_default: dict
    metrics_tuned: dict
    metrics_train: dict
    importances: pd.DataFrame


@dataclass
class Bundle:
    reports: dict[str, ModelReport]
    recommended: str
    X_test: pd.DataFrame
    y_test: np.ndarray
    proba_test: dict[str, np.ndarray]
    reference: pd.DataFrame
    n_train: int
    n_val: int
    n_test: int
    train_positive_rate: float
    test_positive_rate: float


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = beta**2
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def metrics_at(y_true, proba, threshold: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": precision,
        "recall": recall,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "f2": float(fbeta(precision, recall, 2)),
        "matriz": confusion_matrix(y_true, pred, labels=[0, 1]),
    }


def best_threshold(y_true, proba) -> float:
    best_t = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.15, 0.80, 66):
        stats = metrics_at(y_true, proba, float(threshold))
        if stats["f2"] > best_score:
            best_score = stats["f2"]
            best_t = float(threshold)
    return best_t


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ]
    )


def _pretty_feature(name: str) -> str:
    if name.startswith("num__"):
        raw = name.split("__", 1)[1]
        return LABELS.get(raw, raw)
    if name.startswith("cat__"):
        raw = name.split("__", 1)[1]
        for column in CATEGORICAL:
            prefix = f"{column}_"
            if raw.startswith(prefix):
                return f"{LABELS[column]}: {raw[len(prefix):]}"
        return raw
    return name


def _importance_frame(pipeline: Pipeline, kind: str) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    if kind == "coef":
        values = model.coef_.ravel()
    else:
        values = model.feature_importances_
    frame = pd.DataFrame(
        {
            "variable": [_pretty_feature(name) for name in names],
            "valor": values,
        }
    )
    frame["magnitud"] = frame["valor"].abs()
    return frame.sort_values("magnitud", ascending=False).reset_index(drop=True)


def _reference_row(X_train: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for col in NUMERIC:
        row[col] = float(X_train[col].median())
    for col in CATEGORICAL:
        row[col] = X_train[col].mode().iloc[0]
    return pd.DataFrame([row])[FEATURES]


def train_all(df: pd.DataFrame, fast: bool = False) -> Bundle:
    if TARGET not in df.columns:
        raise ModelingError("El dataset no tiene la columna riesgo.")
    if df[TARGET].nunique() < 2:
        raise ModelingError("Hacen falta las dos clases, 0 y 1, para entrenar.")
    if len(df) < 80:
        raise ModelingError("Hacen falta al menos 80 filas para una comparación estable.")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).to_numpy()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    pos = max(int(y_train.sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale = neg / pos
    trees = 80 if fast else 380
    depth = 8 if fast else 12
    boosting_rounds = 70 if fast else 320

    specs = {
        "Regresión logística": (
            Pipeline(
                [
                    ("prep", _preprocessor(scale=True)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            "coef",
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("prep", _preprocessor(scale=False)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=trees,
                            max_depth=depth,
                            min_samples_leaf=4,
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "gain",
        ),
        "XGBoost": (
            Pipeline(
                [
                    ("prep", _preprocessor(scale=False)),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=boosting_rounds,
                            max_depth=3 if fast else 4,
                            learning_rate=0.06,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            min_child_weight=3,
                            reg_lambda=1.2,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=-1,
                            scale_pos_weight=scale,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
            "gain",
        ),
    }

    reports: dict[str, ModelReport] = {}
    proba_test: dict[str, np.ndarray] = {}
    for name, (pipeline, kind) in specs.items():
        pipeline.fit(X_train, y_train)
        p_val = pipeline.predict_proba(X_val)[:, 1]
        p_test = pipeline.predict_proba(X_test)[:, 1]
        p_train = pipeline.predict_proba(X_train)[:, 1]
        threshold = best_threshold(y_val, p_val)
        reports[name] = ModelReport(
            name=name,
            pipeline=pipeline,
            threshold=threshold,
            metrics_default=metrics_at(y_test, p_test, 0.5),
            metrics_tuned=metrics_at(y_test, p_test, threshold),
            metrics_train=metrics_at(y_train, p_train, 0.5),
            importances=_importance_frame(pipeline, kind),
        )
        proba_test[name] = p_test

    recommended = max(
        MODEL_ORDER,
        key=lambda name: (
            reports[name].metrics_tuned["recall"],
            reports[name].metrics_tuned["roc_auc"],
            reports[name].metrics_tuned["f2"],
        ),
    )
    return Bundle(
        reports=reports,
        recommended=recommended,
        X_test=X_test.reset_index(drop=True),
        y_test=y_test,
        proba_test=proba_test,
        reference=_reference_row(X_train),
        n_train=int(len(X_train)),
        n_val=int(len(X_val)),
        n_test=int(len(X_test)),
        train_positive_rate=float(y_train.mean()),
        test_positive_rate=float(y_test.mean()),
    )


def predict_proba(bundle: Bundle, model_name: str, row: pd.DataFrame) -> float:
    pipeline = bundle.reports[model_name].pipeline
    return float(pipeline.predict_proba(row[FEATURES])[0, 1])


def local_factors(bundle: Bundle, model_name: str, row: pd.DataFrame, k: int = 3):
    pipeline = bundle.reports[model_name].pipeline
    base = predict_proba(bundle, model_name, row)
    deltas = []
    for col in FEATURES:
        alt = row.copy()
        alt[col] = bundle.reference[col].values
        other = float(pipeline.predict_proba(alt[FEATURES])[0, 1])
        deltas.append((col, base - other, row.iloc[0][col], bundle.reference.iloc[0][col]))
    increases = sorted([item for item in deltas if item[1] > 0.012], key=lambda item: -item[1])[:k]
    decreases = sorted([item for item in deltas if item[1] < -0.012], key=lambda item: item[1])[:2]
    return base, increases, decreases


def response_curve(bundle: Bundle, model_name: str, column: str, grid: np.ndarray) -> np.ndarray:
    sample = bundle.X_test.sample(n=min(350, len(bundle.X_test)), random_state=42)
    pipeline = bundle.reports[model_name].pipeline
    values = []
    for point in grid:
        tmp = sample.copy()
        tmp[column] = point
        values.append(float(pipeline.predict_proba(tmp[FEATURES])[:, 1].mean()))
    return np.array(values)

"""
Shared feature engineering and price-fit helpers for robot joint modules.

Price is dominated by joint class (QDD vs harmonic vs series-elastic, etc.).
A two-stage fit keeps that structure:

    Stage 1:  log(price) = a[type] + b * log(torque) + c * log(1/accuracy)
    Stage 2:  residual   ~ speed, size, mass, encoder bits, brake, driver

Stage 1 is the data-driven backbone. Stage 2 is a weakly regularized
adjustment so extra sliders move price without flipping the torque sign.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CORE_NUMERIC = ["log_torque", "log_precision"]
RESIDUAL_NUMERIC = [
    "log_speed",
    "log_od",
    "log_mass",
    "encoder_bits",
    "dual_encoder",
    "has_brake",
    "has_driver",
]

# Typical output accuracy (arcmin). Lower is more precise.
TYPE_ACCURACY_ARCMIN = {
    "harmonic": 0.25,
    "cycloidal": 6.0,
    "hobby-servo": 15.0,
    "qdd": 12.0,
    "planetary": 15.0,
    "series-elastic": 15.0,
}

TYPE_ENCODER_BITS = {
    "harmonic": 19,
    "cycloidal": 20,
    "hobby-servo": 12,
    "qdd": 14,
    "planetary": 14,
    "series-elastic": 16,
}


def data_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "data", "robot_joint_data.csv")


def load_joints(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or data_path())


def _parse_encoder(text) -> tuple[float, float]:
    if pd.isna(text):
        return np.nan, 0.0
    raw = str(text).lower()
    dual = 1.0 if "dual" in raw else 0.0
    bits = [int(n) for n in re.findall(r"(\d+)\s*-?\s*bit", raw)]
    if bits:
        return float(max(bits)), dual
    return np.nan, dual


def _brake_flag(text) -> float:
    value = str(text).strip().lower() if pd.notna(text) else "no"
    if value == "yes":
        return 1.0
    if value == "optional":
        return 0.5
    return 0.0


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Torque_Nm"] = out["Peak_Torque_Nm"].fillna(out["Rated_Torque_Nm"])
    parsed = out["Encoder"].apply(_parse_encoder)
    out["encoder_bits"] = [p[0] for p in parsed]
    out["dual_encoder"] = [p[1] for p in parsed]
    out["has_brake"] = out["Brake"].apply(_brake_flag)
    out["has_driver"] = (
        out["Integrated_Driver"].fillna("no").astype(str).str.lower().eq("yes").astype(float)
    )
    out["Accuracy_arcmin"] = out["Type"].map(TYPE_ACCURACY_ARCMIN)
    out["encoder_bits"] = out["encoder_bits"].fillna(out["Type"].map(TYPE_ENCODER_BITS))
    out["log_torque"] = np.log(out["Torque_Nm"])
    out["log_speed"] = np.log(out["Rated_Speed_rpm"])
    out["log_od"] = np.log(out["OD_mm"])
    out["log_mass"] = np.log(out["Weight_kg"])
    out["log_precision"] = np.log(1.0 / out["Accuracy_arcmin"])
    return out


TRAIN_FLAGS = ("listed", "user")


def priced_rows(df: pd.DataFrame, include_estimates: bool = False) -> pd.DataFrame:
    ready = prepare_features(df)
    if "Cost_Flag" in ready.columns and not include_estimates:
        ready = ready[ready["Cost_Flag"].fillna("listed").isin(TRAIN_FLAGS)]
    return ready.dropna(subset=["Cost_USD", "Torque_Nm"]).reset_index(drop=True)


def _core_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), CORE_NUMERIC),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", drop="first"),
                            ["Type"],
                        ),
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )


def _residual_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return Pipeline(
        steps=[
            ("features", ColumnTransformer(transformers=[("num", numeric, RESIDUAL_NUMERIC)])),
            (
                "model",
                ElasticNet(
                    alpha=0.15,
                    l1_ratio=0.2,
                    positive=True,
                    max_iter=10000,
                ),
            ),
        ]
    )


def _mass_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            ["log_torque", "log_od"],
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", drop="first"),
                            ["Type"],
                        ),
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )


@dataclass
class JointPriceModel:
    core_model: Pipeline
    residual_model: Pipeline
    mass_model: Pipeline
    train: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    y_cv: np.ndarray
    core_names: list[str]
    core_coefs: np.ndarray
    core_intercept: float
    residual_names: list[str]
    residual_coefs: np.ndarray
    torque_elasticity: float
    precision_elasticity: float


def _predict_log(core: Pipeline, residual: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    log_core = core.predict(frame[CORE_NUMERIC + ["Type"]])
    log_adj = residual.predict(frame[RESIDUAL_NUMERIC])
    return log_core + log_adj


def fit_models(df: pd.DataFrame | None = None) -> JointPriceModel:
    raw = df if df is not None else load_joints()
    train = priced_rows(raw)
    if len(train) < 8:
        raise ValueError(f"Need more priced joints to fit a model, found {len(train)}")

    y = np.log(train["Cost_USD"].to_numpy(dtype=float))
    core = _core_pipeline()
    core.fit(train[CORE_NUMERIC + ["Type"]], y)
    residual_target = y - core.predict(train[CORE_NUMERIC + ["Type"]])

    residual = _residual_pipeline()
    residual.fit(train[RESIDUAL_NUMERIC], residual_target)

    log_pred = _predict_log(core, residual, train)

    def _clone_fit_predict(train_idx, test_idx):
        core_i = _core_pipeline()
        resid_i = _residual_pipeline()
        core_i.fit(train.loc[train_idx, CORE_NUMERIC + ["Type"]], y[train_idx])
        resid_target = y[train_idx] - core_i.predict(train.loc[train_idx, CORE_NUMERIC + ["Type"]])
        resid_i.fit(train.loc[train_idx, RESIDUAL_NUMERIC], resid_target)
        return _predict_log(core_i, resid_i, train.loc[test_idx])

    loo = LeaveOneOut()
    cv_log = np.zeros_like(y)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        for train_idx, test_idx in loo.split(train):
            cv_log[test_idx] = _clone_fit_predict(train_idx, test_idx)

    mass_train = train.dropna(subset=["Weight_kg"])
    mass_model = _mass_pipeline()
    mass_model.fit(
        mass_train[["log_torque", "log_od", "Type"]],
        np.log(mass_train["Weight_kg"].to_numpy(dtype=float)),
    )

    scaler = core.named_steps["features"].named_transformers_["num"]
    core_lr = core.named_steps["model"]
    torque_elasticity = float(core_lr.coef_[0] / scaler.scale_[0])
    precision_elasticity = float(core_lr.coef_[1] / scaler.scale_[1])

    return JointPriceModel(
        core_model=core,
        residual_model=residual,
        mass_model=mass_model,
        train=train,
        y_true=np.exp(y),
        y_pred=np.exp(log_pred),
        y_cv=np.exp(cv_log),
        core_names=list(core.named_steps["features"].get_feature_names_out()),
        core_coefs=core_lr.coef_,
        core_intercept=float(core_lr.intercept_),
        residual_names=list(residual.named_steps["features"].get_feature_names_out()),
        residual_coefs=residual.named_steps["model"].coef_,
        torque_elasticity=torque_elasticity,
        precision_elasticity=precision_elasticity,
    )


def metrics(fitted: JointPriceModel) -> dict:
    y, pred, cv = fitted.y_true, fitted.y_pred, fitted.y_cv
    return {
        "n_train": int(len(fitted.train)),
        "r2_in_sample": float(r2_score(np.log(y), np.log(pred))),
        "r2_loo": float(r2_score(np.log(y), np.log(cv))),
        "mae_usd": float(mean_absolute_error(y, pred)),
        "mae_loo_usd": float(mean_absolute_error(y, cv)),
        "mape_pct": float(np.mean(np.abs(pred - y) / y) * 100),
        "mape_loo_pct": float(np.mean(np.abs(cv - y) / y) * 100),
        "torque_elasticity": float(fitted.torque_elasticity),
        "precision_elasticity": float(fitted.precision_elasticity),
    }


def coefficient_table(fitted: JointPriceModel) -> pd.DataFrame:
    rows = [{"stage": "core", "feature": "intercept", "log_coef": fitted.core_intercept}]
    for name, coef in zip(fitted.core_names, fitted.core_coefs):
        rows.append({"stage": "core", "feature": name, "log_coef": float(coef)})
    for name, coef in zip(fitted.residual_names, fitted.residual_coefs):
        rows.append({"stage": "residual", "feature": name, "log_coef": float(coef)})
    table = pd.DataFrame(rows)
    table["approx_x_factor"] = np.exp(table["log_coef"])
    table.loc[table["feature"] == "intercept", "approx_x_factor"] = np.nan
    return table


def _spec_frame(
    torque_nm: float,
    speed_rpm: float,
    od_mm: float,
    mass_kg: float | None,
    accuracy_arcmin: float,
    joint_type: str,
    encoder_bits: float,
    dual_encoder: bool,
    has_brake: bool,
    has_driver: bool,
    fitted: JointPriceModel,
) -> tuple[pd.DataFrame, float]:
    if mass_kg is None:
        seed = pd.DataFrame(
            {
                "log_torque": [np.log(max(torque_nm, 0.05))],
                "log_od": [np.log(max(od_mm, 10.0))],
                "Type": [joint_type],
            }
        )
        mass_kg = float(np.exp(fitted.mass_model.predict(seed)[0]))

    frame = pd.DataFrame(
        {
            "log_torque": [np.log(max(torque_nm, 0.05))],
            "log_speed": [np.log(max(speed_rpm, 1.0))],
            "log_od": [np.log(max(od_mm, 10.0))],
            "log_mass": [np.log(max(mass_kg, 0.02))],
            "log_precision": [np.log(1.0 / max(accuracy_arcmin, 0.05))],
            "encoder_bits": [encoder_bits],
            "dual_encoder": [1.0 if dual_encoder else 0.0],
            "has_brake": [1.0 if has_brake else 0.0],
            "has_driver": [1.0 if has_driver else 0.0],
            "Type": [joint_type],
        }
    )
    return frame, float(mass_kg)


def estimate_price(
    fitted: JointPriceModel,
    torque_nm: float,
    speed_rpm: float,
    od_mm: float,
    mass_kg: float | None = None,
    accuracy_arcmin: float | None = None,
    joint_type: str = "qdd",
    encoder_bits: float | None = None,
    dual_encoder: bool = True,
    has_brake: bool = False,
    has_driver: bool = True,
) -> dict:
    accuracy = (
        TYPE_ACCURACY_ARCMIN.get(joint_type, 12.0)
        if accuracy_arcmin is None
        else accuracy_arcmin
    )
    bits = TYPE_ENCODER_BITS.get(joint_type, 14.0) if encoder_bits is None else encoder_bits
    X, used_mass = _spec_frame(
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        od_mm=od_mm,
        mass_kg=mass_kg,
        accuracy_arcmin=accuracy,
        joint_type=joint_type,
        encoder_bits=bits,
        dual_encoder=dual_encoder,
        has_brake=has_brake,
        has_driver=has_driver,
        fitted=fitted,
    )
    log_price = float(_predict_log(fitted.core_model, fitted.residual_model, X)[0])
    price = float(np.clip(np.exp(log_price), 20.0, 50_000.0))
    return {
        "cost_usd": price,
        "mass_kg": used_mass,
        "accuracy_arcmin": float(accuracy),
        "encoder_bits": float(bits),
        "mass_was_estimated": mass_kg is None,
    }

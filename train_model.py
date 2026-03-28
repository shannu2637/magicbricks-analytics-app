"""
train_model.py
--------------
Run once to train and save the model:
    python train_model.py --data MagicBrick_Data.csv

Outputs:
    model.pkl          - trained sklearn pipeline
    model_meta.json    - feature lists + metrics
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor


def parse_price(v):
    v = str(v).strip().replace("₹", "").strip()
    if "Cr" in v:
        return float(v.replace("Cr", "").strip()) * 1e7
    elif "Lac" in v:
        return float(v.replace("Lac", "").strip()) * 1e5
    try:
        return float(v)
    except Exception:
        return float("nan")


def parse_emi(v):
    v = str(v).strip()
    if "k" in v:
        return float(v.replace("k", "").strip()) * 1e3
    elif "L" in v:
        return float(v.replace("L", "").strip()) * 1e5
    try:
        return float(v)
    except Exception:
        return float("nan")


def parse_area(v):
    v = str(v).lower().replace(",", "").strip()
    if v == "nan":
        return float("nan")
    if v.replace(".", "").isdigit():
        return float(v)
    if "sqm" in v:
        return float(v.replace("sqm", "").strip()) * 10.7639
    if "sqyrd" in v:
        return float(v.replace("sqyrd", "").strip()) * 9
    return float("nan")


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    if df["Price"].dtype == object:
        df["Price"] = df["Price"].apply(parse_price)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    if "EMI" in df.columns and df["EMI"].dtype == object:
        df["EMI"] = df["EMI"].apply(parse_emi)

    if df["Carpet Area"].dtype == object:
        df["Carpet Area"] = df["Carpet Area"].apply(parse_area)
    df["Carpet Area"] = pd.to_numeric(df["Carpet Area"], errors="coerce")

    obj_cols = df.select_dtypes(include="object").columns
    df.dropna(subset=obj_cols, inplace=True)
    room_cols = [c for c in ["BHK", "Bathrooms", "Balconies"] if c in df.columns]
    df.dropna(subset=room_cols, inplace=True)
    df.drop_duplicates(inplace=True)

    if "Locality" in df.columns:
        for col in ["Price", "EMI"]:
            if col in df.columns:
                df[col] = df[col].fillna(df.groupby("Locality")[col].transform("median"))

    df.dropna(subset=["Price", "Carpet Area"], inplace=True)
    df = df[df["Price"] > 0]

    df["Carpet_to_bhk"]  = df["Carpet Area"] / df["BHK"].replace(0, np.nan)
    df["Carpet_to_bath"] = df["Carpet Area"] / df["Bathrooms"].replace(0, np.nan)
    df["log_carpet"]     = np.log1p(df["Carpet Area"])

    return df.reset_index(drop=True)


def build_pipeline(cat_features, num_features):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
        ("num", StandardScaler(), num_features),
    ])
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def train(data_path: str, out_model: str = "model.pkl", out_meta: str = "model_meta.json"):
    print(f"Loading data from: {data_path}")
    df = load_and_clean(data_path)
    print(f"Clean records: {len(df)}")

    cat_features = [c for c in ["City", "Locality", "Developer"] if c in df.columns]
    num_features = [c for c in [
        "BHK", "Bathrooms", "Balconies", "Carpet Area",
        "Carpet_to_bhk", "Carpet_to_bath", "log_carpet"
    ] if c in df.columns]

    X = df[cat_features + num_features]
    y = np.log1p(df["Price"])

    pipe = build_pipeline(cat_features, num_features)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2   = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    cv_rmse = np.sqrt(-cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error"))

    pipe.fit(X, y)

    y_pred_log = pipe.predict(X)
    y_true = np.expm1(y.values)
    y_pred = np.expm1(y_pred_log)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n--- Metrics ---")
    print(f"Train R²:    {r2:.4f}")
    print(f"CV R² mean:  {cv_r2.mean():.4f}  ± {cv_r2.std():.4f}")
    print(f"MAE:         ₹{mae/1e5:.2f} Lac")
    print(f"RMSE:        ₹{rmse/1e7:.2f} Cr")

    joblib.dump(pipe, out_model)
    print(f"\nModel saved → {out_model}")

    meta = {
        "cat_features": cat_features,
        "num_features": num_features,
        "cities":       sorted(df["City"].unique().tolist()),
        "localities":   sorted(df["Locality"].unique().tolist()),
        "developers":   sorted(df["Developer"].unique().tolist()),
        "metrics": {
            "train_r2":   round(r2, 4),
            "cv_r2_mean": round(cv_r2.mean(), 4),
            "cv_r2_std":  round(cv_r2.std(), 4),
            "mae_lac":    round(mae / 1e5, 2),
            "rmse_cr":    round(rmse / 1e7, 2),
        },
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved → {out_meta}")
    return pipe, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="MagicBrick_Data.csv")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--meta",  default="model_meta.json")
    args = parser.parse_args()
    train(args.data, args.model, args.meta)

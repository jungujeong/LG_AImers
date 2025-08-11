#%%
import os
import random
import glob
import re

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Optional dependencies for optimization and GBDTs
try:
    import optuna
except Exception:
    optuna = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


#%%
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)


#%%
# Hyperparameters
LOOKBACK = 28
PREDICT = 7

# RandomForest 기본 설정 (필요 시 조정)
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 16,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
}


#%%
train = pd.read_csv("./train/train.csv")


#%%
def compute_window_stats(window_values: np.ndarray) -> np.ndarray:
    """
    주어진 LOOKBACK 길이의 1차원 window에서 추가 통계 피처를 계산한다.
    포함 피처:
      - rolling mean(7/14/28)
      - rolling std(7/14/28)
      - 최근 7일 합과 이전 7일 합의 차이(최근 추세 근사)
    반환: shape (10,) 의 벡터
    """
    assert window_values.ndim == 1 and len(window_values) == LOOKBACK

    last7 = window_values[-7:]
    last14 = window_values[-14:]
    last28 = window_values[-28:]

    mean7 = float(np.mean(last7))
    mean14 = float(np.mean(last14))
    mean28 = float(np.mean(last28))

    std7 = float(np.std(last7))
    std14 = float(np.std(last14))
    std28 = float(np.std(last28))

    # 최근 7일 합 - 그 이전 7일 합 (증감 추세 근사)
    prev7 = window_values[-14:-7]
    trend7 = float(np.sum(last7) - np.sum(prev7))

    # 마지막 관측치와 7일 평균의 차이
    last_value = float(window_values[-1])
    last_minus_mean7 = last_value - mean7

    return np.array([mean7, mean14, mean28, std7, std14, std28, trend7, last_value, last_minus_mean7, float(np.max(last7))])


def build_supervised_series(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    단일 시계열에서 학습용 (X, y) 샘플을 생성한다.
    X: [lags 28개 + 통계피처 10개] -> 총 38차원
    y: 다음 7일 매출 (멀티아웃풋)
    """
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    # series 길이 검증
    if len(series) < LOOKBACK + PREDICT:
        return np.empty((0, LOOKBACK + 10), dtype=float), np.empty((0, PREDICT), dtype=float)

    for end in range(LOOKBACK, len(series) - PREDICT + 1):
        window = series[end - LOOKBACK : end]
        future = series[end : end + PREDICT]

        lags = window.astype(float)
        stats = compute_window_stats(window)
        X_feat = np.concatenate([lags, stats], axis=0)
        X_list.append(X_feat)
        Y_list.append(future.astype(float))

    return np.vstack(X_list), np.vstack(Y_list)


#%%
def train_rf_models(train_df: pd.DataFrame) -> dict:
    """업장명_메뉴명 단위로 RandomForest Regressor(멀티아웃풋)를 학습한다."""
    trained: dict = {}

    for key, group in tqdm(train_df.groupby(["영업장명_메뉴명"]), desc="Training RF"):
        group_sorted = group.sort_values("영업일자").copy()
        series = group_sorted["매출수량"].values

        X, y = build_supervised_series(series)
        if X.shape[0] == 0:
            continue

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X, y)

        # 예측 시 사용할 마지막 윈도우 저장
        last_window = series[-LOOKBACK:]
        trained[key] = {
            "model": model,
            "last_window": last_window,
        }

    return trained


#%%
def predict_rf_for_group(model: RandomForestRegressor, last_window: np.ndarray) -> np.ndarray:
    """단일 그룹에서 마지막 관측 윈도우로 다음 7일을 예측한다."""
    lags = last_window.astype(float)
    stats = compute_window_stats(last_window)
    X = np.concatenate([lags, stats], axis=0)[None, :]
    pred = model.predict(X).reshape(-1)
    pred = np.clip(pred, 0.0, None)  # 음수 방지
    return pred


#%%
def predict_rf(test_df: pd.DataFrame, trained_models: dict, test_prefix: str) -> pd.DataFrame:
    results: list[dict] = []

    for key, store_test in test_df.groupby(["영업장명_메뉴명"]):
        if key not in trained_models:
            continue

        model = trained_models[key]["model"]

        store_test_sorted = store_test.sort_values("영업일자")
        recent_vals = store_test_sorted["매출수량"].values[-LOOKBACK:]
        if len(recent_vals) < LOOKBACK:
            continue

        preds = predict_rf_for_group(model, recent_vals)

        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]
        for d, val in zip(pred_dates, preds):
            results.append({
                "영업일자": d,
                "영업장명_메뉴명": key,
                "매출수량": float(val),
            })

    return pd.DataFrame(results)


#%%
def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame) -> pd.DataFrame:
    """
    예측 결과를 제출 파일 포맷으로 변환한다.
    주의: 사용자가 검증한 매핑을 유지한다: (date, (col,))
    """
    pred_dict = dict(
        zip(zip(pred_df["영업일자"], pred_df["영업장명_메뉴명"]), pred_df["매출수량"])
    )

    final_df = sample_submission.copy()
    # Avoid dtype warnings: cast prediction columns to float
    final_df.iloc[:, 1:] = final_df.iloc[:, 1:].astype("float64")
    for row_idx in final_df.index:
        date = final_df.loc[row_idx, "영업일자"]
        for col in final_df.columns[1:]:
            final_df.loc[row_idx, col] = pred_dict.get((date, (col,)), 0)
    return final_df


#%%
def run_all(output_csv: str = "baseline_ml_submission.csv") -> None:
    # 1) 학습
    trained_models = train_rf_models(train)

    # 2) 예측
    all_preds: list[pd.DataFrame] = []
    test_files = sorted(glob.glob("./test/TEST_*.csv"))
    for path in test_files:
        test_df = pd.read_csv(path)
        filename = os.path.basename(path)
        test_prefix = re.search(r"(TEST_\d+)", filename).group(1)
        pred_df = predict_rf(test_df, trained_models, test_prefix)
        all_preds.append(pred_df)

    full_pred_df = pd.concat(all_preds, ignore_index=True)

    # 3) 제출 변환 및 저장
    sample_submission = pd.read_csv("./sample_submission.csv")
    submission = convert_to_submission_format(full_pred_df, sample_submission)
    submission.to_csv(output_csv, index=False, encoding="utf-8-sig")


## NOTE: 엔트리포인트는 모든 함수 정의 뒤에 위치해야 함



##############################
# Optuna-based model selection
##############################

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Compute SMAPE on flattened arrays, excluding y_true == 0."""
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mask = yt != 0
    if not np.any(mask):
        return 0.0
    yt = yt[mask]
    yp = yp[mask]
    return float(np.mean(2.0 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + eps)))


def split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2):
    n = X.shape[0]
    n_val = max(1, int(n * val_ratio))
    n_tr = max(1, n - n_val)
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def make_estimator(model_type: str, params: dict):
    if model_type == "rf":
        return RandomForestRegressor(**params)
    if model_type == "lgbm":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm가 설치되어 있지 않습니다. pip install lightgbm")
        return MultiOutputRegressor(LGBMRegressor(**params))
    if model_type == "cat":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost가 설치되어 있지 않습니다. pip install catboost")
        return MultiOutputRegressor(CatBoostRegressor(**params))
    raise ValueError(f"unknown model_type: {model_type}")


def build_group_datasets(train_df: pd.DataFrame):
    """Build (X, y) datasets per group for validation/optuna usage."""
    datasets = {}
    for key, group in train_df.groupby(["영업장명_메뉴명"]):
        group_sorted = group.sort_values("영업일자").copy()
        series = group_sorted["매출수량"].values
        X, y = build_supervised_series(series)
        if X.shape[0] == 0:
            continue
        datasets[key] = {"X": X, "y": y}
    return datasets


def eval_model_on_validation(model_type: str, params: dict, datasets: dict, val_ratio: float = 0.2) -> float:
    preds_all = []
    trues_all = []
    for key, d in datasets.items():
        X, y = d["X"], d["y"]
        if X.shape[0] < 5:
            continue
        X_tr, y_tr, X_val, y_val = split_train_val(X, y, val_ratio)
        try:
            est = make_estimator(model_type, params)
            est.fit(X_tr, y_tr)
            y_hat = est.predict(X_val)
        except Exception:
            continue
        preds_all.append(y_hat)
        trues_all.append(y_val)
    if not preds_all:
        return 1e9
    y_pred = np.vstack(preds_all)
    y_true = np.vstack(trues_all)
    return smape(y_true, y_pred)


def optimize_hyperparams(train_df: pd.DataFrame, n_trials: int = 20, val_ratio: float = 0.2):
    if optuna is None:
        raise RuntimeError("optuna가 설치되어 있지 않습니다. pip install optuna")
    datasets = build_group_datasets(train_df)

    best_params = {}
    studies = {}

    def obj_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 8, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "n_jobs": -1,
            "random_state": 42,
        }
        return eval_model_on_validation("rf", params, datasets, val_ratio)

    def obj_lgbm(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 24),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1,
        }
        return eval_model_on_validation("lgbm", params, datasets, val_ratio)

    def obj_cat(trial):
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": 0,
        }
        return eval_model_on_validation("cat", params, datasets, val_ratio)

    for name, obj in [("rf", obj_rf), ("lgbm", obj_lgbm), ("cat", obj_cat)]:
        try:
            study = optuna.create_study(direction="minimize")
            study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
            studies[name] = study
            best_params[name] = study.best_params
        except Exception as e:
            print(f"[optuna] {name} 최적화 실패: {e}")

    return best_params, studies


def train_models_generic(train_df: pd.DataFrame, model_type: str, params: dict) -> dict:
    trained = {}
    for key, group in tqdm(train_df.groupby(["영업장명_메뉴명"]), desc=f"Training {model_type}"):
        group_sorted = group.sort_values("영업일자").copy()
        series = group_sorted["매출수량"].values
        X, y = build_supervised_series(series)
        if X.shape[0] == 0:
            continue
        est = make_estimator(model_type, params)
        est.fit(X, y)
        last_window = series[-LOOKBACK:]
        trained[key] = {"model": est, "last_window": last_window}
    return trained


def predict_generic(test_df: pd.DataFrame, trained_models: dict, test_prefix: str) -> pd.DataFrame:
    results: list[dict] = []
    for key, store_test in test_df.groupby(["영업장명_메뉴명"]):
        if key not in trained_models:
            continue
        model = trained_models[key]["model"]
        store_test_sorted = store_test.sort_values("영업일자")
        recent_vals = store_test_sorted["매출수량"].values[-LOOKBACK:]
        if len(recent_vals) < LOOKBACK:
            continue
        lags = recent_vals.astype(float)
        stats = compute_window_stats(recent_vals)
        X = np.concatenate([lags, stats], axis=0)[None, :]
        preds = model.predict(X).reshape(-1)
        preds = np.clip(preds, 0.0, None)
        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]
        for d, val in zip(pred_dates, preds):
            results.append({
                "영업일자": d,
                "영업장명_메뉴명": key,
                "매출수량": float(val),
            })
    return pd.DataFrame(results)


def optimize_ensemble_weights(val_preds: dict, y_true: np.ndarray, n_trials: int = 100):
    if optuna is None:
        raise RuntimeError("optuna가 설치되어 있지 않습니다. pip install optuna")
    keys = list(val_preds.keys())
    mats = [val_preds[k] for k in keys]

    def objective(trial):
        ws = [trial.suggest_float(f"w_{k}", 0.0, 1.0) for k in keys]
        s = sum(ws) + 1e-12
        ws = [w / s for w in ws]
        y_hat = np.zeros_like(y_true, dtype=float)
        for w, m in zip(ws, mats):
            y_hat += w * m
        return smape(y_true, y_hat)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    norm = sum(best.values()) + 1e-12
    best = {k: v / norm for k, v in best.items()}
    return best, study


def run_optuna_pipeline(n_trials: int = 20, val_ratio: float = 0.2, output_csv: str = "baseline_ml_submission.csv"):
    """Run hyperparameter optimization for RF/LGBM/Cat, compare ensemble vs best single, and export submission."""
    best_params, studies = optimize_hyperparams(train, n_trials=n_trials, val_ratio=val_ratio)
    print("[best params]", best_params)

    datasets = build_group_datasets(train)
    val_preds = {}
    y_true_mat = None

    for model_type in ["rf", "lgbm", "cat"]:
        if model_type not in best_params:
            continue
        params = best_params[model_type]
        preds_all = []
        trues_all = []
        for key, d in datasets.items():
            X, y = d["X"], d["y"]
            if X.shape[0] < 5:
                continue
            X_tr, y_tr, X_val, y_val = split_train_val(X, y, val_ratio=val_ratio)
            est = make_estimator(model_type, params)
            est.fit(X_tr, y_tr)
            y_hat = est.predict(X_val)
            preds_all.append(y_hat)
            trues_all.append(y_val)
        if preds_all:
            val_preds[model_type] = np.vstack(preds_all)
            if y_true_mat is None:
                y_true_mat = np.vstack(trues_all)

    if not val_preds:
        raise RuntimeError("validation predictions are empty")

    best_weights, _ = optimize_ensemble_weights(val_preds, y_true_mat, n_trials=100)
    print("[best ensemble weights]", best_weights)

    single_scores = {k: smape(y_true_mat, v) for k, v in val_preds.items()}
    ens_pred = np.zeros_like(y_true_mat)
    for k, v in val_preds.items():
        ens_pred += best_weights.get(f"w_{k}", 0.0) * v
    ens_score = smape(y_true_mat, ens_pred)
    print("[single scores]", single_scores, "[ensemble]", ens_score)

    if ens_score < min(single_scores.values()):
        choice = "ensemble"
        chosen = list(val_preds.keys())
    else:
        choice = min(single_scores, key=single_scores.get)
        chosen = [choice]
    print(f"[final choice] {choice}")

    # Train final models on full data
    trained_models_dict = {}
    for model_type in chosen:
        params = best_params[model_type]
        models = train_models_generic(train, model_type, params)
        trained_models_dict[model_type] = models

    # Predict on tests
    all_preds = []
    test_files = sorted(glob.glob("./test/TEST_*.csv"))
    for path in test_files:
        test_df = pd.read_csv(path)
        filename = os.path.basename(path)
        test_prefix = re.search(r"(TEST_\d+)", filename).group(1)
        if choice == "ensemble":
            per_model = []
            for model_type in chosen:
                dfp = predict_generic(test_df, trained_models_dict[model_type], test_prefix)
                dfp = dfp.sort_values(["영업일자", "영업장명_메뉴명"]).reset_index(drop=True)
                per_model.append((model_type, dfp))
            base = per_model[0][1].copy()
            base.rename(columns={"매출수량": f"pred_{per_model[0][0]}"}, inplace=True)
            for model_type, dfp in per_model[1:]:
                base = base.merge(dfp, on=["영업일자", "영업장명_메뉴명"], how="left", suffixes=("", f"_{model_type}"))
                base.rename(columns={"매출수량": f"pred_{model_type}"}, inplace=True)
            pred_cols = [c for c in base.columns if c.startswith("pred_")]
            base["매출수량"] = 0.0
            for c in pred_cols:
                w = best_weights.get(f"w_{c.split('_', 1)[1]}", 0.0)
                base["매출수량"] += w * base[c]
            final_pred = base[["영업일자", "영업장명_메뉴명", "매출수량"]]
        else:
            final_pred = predict_generic(test_df, trained_models_dict[choice], test_prefix)
        all_preds.append(final_pred)

    full_pred_df = pd.concat(all_preds, ignore_index=True)
    sample_submission = pd.read_csv("./sample_submission.csv")
    submission = convert_to_submission_format(full_pred_df, sample_submission)
    submission.to_csv(output_csv, index=False, encoding="utf-8-sig")


#%%
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "rf"
    if mode.lower() == "optuna":
        if optuna is None:
            print("[warn] optuna가 설치되어 있지 않습니다. pip install optuna lightgbm catboost 후 다시 실행하세요.")
            sys.exit(1)
        print("[run] Optuna 파이프라인 실행 → baseline_ml_optuna.csv")
        run_optuna_pipeline(n_trials=30, val_ratio=0.2, output_csv="baseline_ml_optuna.csv")
    else:
        print("[run] RF 기본 파이프라인 실행 → baseline_ml_submission.csv")
        run_all(output_csv="baseline_ml_submission.csv")

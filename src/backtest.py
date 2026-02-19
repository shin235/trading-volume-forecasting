import numpy as np
import pandas as pd
import tensorflow as tf
from .config import SEED, set_seeds
from .data import apply_split_residual, build_sample_index, make_case_residual, split_samples_by_target_time, fit_x_scaler, make_case_A, make_case_B, make_case_C, apply_split_single, apply_split_two_now, apply_split_two_seq
from .models import predict_persistence, build_gru_delta, build_gru_single, build_gru_two_branch_now, build_gru_two_branch_seq
from .training import train_single, train_two_now, train_two_seq
from .evaluation import eval_predictions


def rolling_origin_splits(n, initial_train_size, test_block_size, val_size):
    """
    Rolling-origin backtest splits with an expanding training window.

    For each fold:
      - train rows: [0 .. train_end]
      - val rows:   [train_end - val_size + 1 .. train_end]  (last part of train period)
      - test rows:  [train_end + 1 .. train_end + test_block_size]

    The window advances by one test block each fold.
    """
    train_end = initial_train_size - 1

    while True:
        test_start = train_end + 1
        test_end = train_end + test_block_size
        if test_end >= n:
            break

        train_rows = np.arange(0, train_end + 1, dtype=np.int32)

        val_start = max(0, train_end - val_size + 1)
        val_rows = np.arange(val_start, train_end + 1, dtype=np.int32)

        test_rows = np.arange(test_start, test_end + 1, dtype=np.int32)

        yield train_rows, val_rows, test_rows, train_end

        train_end += test_block_size


def run_backtest(
    df,
    X_raw,
    y_model,
    window,
    horizon,
    use_log1p,
    case_name,
    initial_train_size,
    test_block_size,
    val_size,
    plot_fold=0,
    return_artifacts=False,
    seed=SEED,
):
    """
    Run rolling-origin backtesting for one case (P/A/B/C).

    Key points for leakage-safe evaluation:
    - Samples are split by target time (t+horizon), so train/val/test targets are mutually exclusive.
    - The X-scaler is fit on pure training rows only (validation excluded), so preprocessing cannot peek.
    - Optionally returns artifacts (loss curve + predictions) for a single fold to make plots reproducible.
    """
    if case_name not in ["P", "A", "B", "C"]:
        raise ValueError("case_name must be one of: P, A, B, C")

    n = len(df)
    t_index_all, target_index_all = build_sample_index(n, window, horizon)

    artifacts = {}
    fold_metrics = []

    for fold, (train_rows, val_rows, test_rows, train_end) in enumerate(
        rolling_origin_splits(n, initial_train_size, test_block_size, val_size)
    ):
        # reproducible randomness per fold
        set_seeds(seed + fold)

        # 1) Build mutually exclusive masks by target time
        tr_mask, va_mask, te_mask = split_samples_by_target_time(
            target_index_all, train_rows, val_rows, test_rows
        )

        # 2) Fit X scaler on pure train rows only (exclude val rows)
        train_rows_fit = np.setdiff1d(train_rows, val_rows).astype(np.int32)
        X_scaled, _ = fit_x_scaler(X_raw, train_rows_fit)

        # 3) Build dataset + train + predict
        if case_name == "A":
            X, y = make_case_A(y_model, t_index_all, target_index_all, window)
            (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = apply_split_single(
                X, y, tr_mask, va_mask, te_mask
            )

            model = build_gru_single(window, n_in=1)
            history = train_single(
                model, X_tr, y_tr, X_va, y_va, epochs=200, batch_size=32)
            pred = model.predict(X_te, verbose=0).reshape(-1)

            if return_artifacts and fold == plot_fold:
                artifacts["A_history"] = history.history
                artifacts["A_y_true_model"] = y_te.copy()
                artifacts["A_y_pred_model"] = pred.copy()
                artifacts["A_target_index"] = target_index_all[te_mask].copy()

        elif case_name == "B":
            (Xy, Xc), y = make_case_B(y_model, X_scaled,
                                      t_index_all, target_index_all, window)
            (X_tr_in, y_tr), (X_va_in, y_va), (X_te_in, y_te) = apply_split_two_now(
                Xy, Xc, y, tr_mask, va_mask, te_mask
            )

            model = build_gru_two_branch_now(
                window, n_feat_now=X_scaled.shape[1])
            history = train_two_now(
                model, X_tr_in, y_tr, X_va_in, y_va, epochs=200, batch_size=32)
            pred = model.predict(X_te_in, verbose=0).reshape(-1)

        elif case_name == "C":
            (Xy, Xh), y = make_case_C(y_model, X_scaled,
                                      t_index_all, target_index_all, window)
            (X_tr_in, y_tr), (X_va_in, y_va), (X_te_in, y_te) = apply_split_two_seq(
                Xy, Xh, y, tr_mask, va_mask, te_mask
            )

            model = build_gru_two_branch_seq(
                window_y=window, window_x=window - 1, n_feat_x=X_scaled.shape[1]
            )
            history = train_two_seq(
                model, X_tr_in, y_tr, X_va_in, y_va, epochs=200, batch_size=32)
            pred = model.predict(X_te_in, verbose=0).reshape(-1)

        elif case_name == "P":
            # true targets for all samples (by target time)
            y_all = y_model[target_index_all].astype(np.float32)

            # persistence: y_hat_{t+h} = y_t  (here h=1)
            pred_all = predict_persistence(y_model, t_index_all)

            y_te = y_all[te_mask]
            pred = pred_all[te_mask]

            if return_artifacts and fold == plot_fold:
                artifacts["P_y_true_model"] = y_te.copy()
                artifacts["P_y_pred_model"] = pred.copy()
                artifacts["P_target_index"] = target_index_all[te_mask].copy()

        # sanity checks
        assert len(t_index_all) == len(target_index_all)
        assert len(y_te) == len(pred)

        # 4) Evaluate
        metrics = eval_predictions(y_te, pred, use_log1p=use_log1p)
        metrics["fold"] = fold
        metrics["train_end_row"] = int(train_end)
        metrics["n_test_samples"] = int(len(y_te))
        fold_metrics.append(metrics)

        tf.keras.backend.clear_session()

    df_metrics = pd.DataFrame(fold_metrics)
    if return_artifacts:
        return df_metrics, artifacts
    return df_metrics


def run_backtest_residual(
    df,
    y_model,
    window,
    horizon,
    use_log1p,
    initial_train_size,
    test_block_size,
    val_size,
    case_name,                 # "P" or "Res"
    plot_fold=0,
    return_artifacts=False,
    seed=SEED,
    epochs=100,
    batch_size=32,
):
    """
    Run rolling-origin backtesting for residual setting (P vs Res).

    Returns per-fold metrics on:
      - model scale (log1p space): RMSE_model, MAE_model, R2_model
      - original scale:            RMSE_orig, MAE_orig

    Leakage-safe points:
    - Splits are done by target time (t+h).
    - Residual target uses only y_{t} and y_{t+h}.
    - Optionally returns artifacts for one fold for reproducible plots.
    """
    if case_name not in ["P", "Res"]:
        raise ValueError("case_name must be one of: P, Res")

    n = len(df)
    t_index_all, target_index_all = build_sample_index(n, window, horizon)

    artifacts = {}
    fold_metrics = []

    for fold, (train_rows, val_rows, test_rows, train_end) in enumerate(
        rolling_origin_splits(n, initial_train_size, test_block_size, val_size)
    ):
        set_seeds(seed + fold)

        # leakage-safe masks by target time
        tr_mask, va_mask, te_mask = split_samples_by_target_time(
            target_index_all, train_rows, val_rows, test_rows
        )

        # residual dataset
        Xy_all, r_all, y_t_all, y_next_all = make_case_residual(
            y_model, t_index_all, target_index_all, window
        )

        (X_tr, r_tr, y_t_tr, y_next_tr), (X_va, r_va, y_t_va, y_next_va), (X_te, r_te, y_t_te, y_next_te) = apply_split_residual(
            Xy_all, r_all, y_t_all, y_next_all, tr_mask, va_mask, te_mask
        )

        if case_name == "P":
            # persistence: y_hat_{t+h} = y_t  (here h=1)
            y_pred_model = y_t_te.astype(np.float32)
            y_true_model = y_next_te.astype(np.float32)

            if return_artifacts and fold == plot_fold:
                artifacts["P_y_true_model"] = y_true_model.copy()
                artifacts["P_y_pred_model"] = y_pred_model.copy()
                artifacts["P_target_index"] = target_index_all[te_mask].copy()

        else:
            # residual GRU
            model = build_gru_delta(window=window)
            history = train_single(
                model, X_tr, r_tr, X_va, r_va, epochs=epochs, batch_size=batch_size
            )

            r_hat = model.predict(
                X_te, verbose=0).reshape(-1).astype(np.float32)
            y_pred_model = (y_t_te + r_hat).astype(np.float32)
            y_true_model = y_next_te.astype(np.float32)

            if return_artifacts and fold == plot_fold:
                artifacts["Res_history"] = history.history
                artifacts["Res_y_true_model"] = y_true_model.copy()
                artifacts["Res_y_pred_model"] = y_pred_model.copy()
                artifacts["Res_target_index"] = target_index_all[te_mask].copy()

        # evaluate on both scales
        metrics = eval_predictions(
            y_true_model, y_pred_model, use_log1p=use_log1p)
        metrics["fold"] = fold
        metrics["train_end_row"] = int(train_end)
        metrics["n_test_samples"] = int(len(y_true_model))
        fold_metrics.append(metrics)

        tf.keras.backend.clear_session()

    df_metrics = pd.DataFrame(fold_metrics)
    if return_artifacts:
        return df_metrics, artifacts
    return df_metrics

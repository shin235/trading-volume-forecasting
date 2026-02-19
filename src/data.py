import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. loading and preprocessing


def load_time_series_csv(use_log1p=True):
    df = pd.read_csv("Time_Series.csv")

    # first column = date, last column = target
    date_col = df.columns[0]
    target_col = df.columns[-1]

    # covariates: numeric columns except date and target
    cov_cols = [
        c for c in df.columns
        if c not in [date_col, target_col]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X_raw = df[cov_cols].astype(np.float32).to_numpy() if cov_cols else None
    y_raw = df[target_col].astype(np.float32).to_numpy()

    # sanity checks
    if np.isnan(y_raw).any():
        raise ValueError("y has NaNs. Please clean the dataset.")
    if X_raw is not None and np.isnan(X_raw).any():
        raise ValueError("X has NaNs. Please clean the dataset.")

    # target transform
    if use_log1p:
        y_model = np.log1p(y_raw).astype(np.float32)
    else:
        y_model = y_raw.astype(np.float32)

    return df, X_raw, y_raw, y_model, cov_cols, date_col, target_col


# 2. indexing
def build_sample_index(n, window, horizon):
    t_min = window - 1
    t_max = n - 1 - horizon
    t_index = np.arange(t_min, t_max + 1, dtype=np.int32)
    target_index = t_index + horizon
    return t_index, target_index


def split_samples_by_target_time(target_index, train_rows, val_rows, test_rows):

    te_mask = np.isin(target_index, test_rows)
    va_mask = np.isin(target_index, val_rows)
    tr_mask = np.isin(target_index, train_rows) & ~va_mask & ~te_mask
    return tr_mask, va_mask, te_mask


# 3. scaling
def fit_x_scaler(X_raw, train_rows_fit):
    if X_raw is None:
        return None, None
    x_scaler = StandardScaler()
    x_scaler.fit(X_raw[train_rows_fit])
    X_scaled = x_scaler.transform(X_raw).astype(np.float32)
    return X_scaled, x_scaler


# 4. input builders
def make_case_A(y_seq, t_index, target_index, window):
    """
    Model A: y_{t+1} = f(y_{t-w+1:t}). Uses only the target history.
    Xy shape: (N, window, 1)
    """
    Xy = np.stack([y_seq[t - window + 1: t + 1].reshape(window, 1)
                  for t in t_index]).astype(np.float32)
    yt = y_seq[target_index].astype(np.float32)
    return Xy, yt


def make_case_B(y_seq, X_scaled, t_index, target_index, window):
    """
    Model B: y_{t+1} = f(y_{t-w+1:t}, X_{t-1})
    Uses target history and the most recent covariate vector.
    """
    if X_scaled is None:
        raise ValueError("No covariates available for B.")

    Xy = np.stack([y_seq[t - window + 1: t + 1].reshape(window, 1)
                  for t in t_index]).astype(np.float32)
    Xc = X_scaled[t_index - 1].astype(np.float32)  # X_{t-1}
    yt = y_seq[target_index].astype(np.float32)
    return (Xy, Xc), yt


def make_case_C(y_seq, X_scaled, t_index, target_index, window):
    """
    Model C: y_{t+1} = f(y_{t-w+1:t}, X_{t-w+1:t-1})
    Uses both target history and covariate history.
    Implemented as two sequence inputs.
    """
    if X_scaled is None:
        raise ValueError("No covariates available for C.")

    Xy = np.stack([y_seq[t - window + 1: t + 1].reshape(window, 1)
                  for t in t_index]).astype(np.float32)

    # X-history length = window-1 (up to t-1)
    Xh = np.stack([X_scaled[t - window + 1: t] for t in t_index]
                  ).astype(np.float32)  # shape (N, window-1, dX)

    yt = y_seq[target_index].astype(np.float32)
    return (Xy, Xh), yt


def make_case_residual(y_model, t_index, target_index, window):
    """
    Inputs:
      Xy[t] = [y_{t-w+1}, ..., y_t] as (window, 1)
    Targets:
      r_{t+1} = y_{t+1} - y_t
    """
    Xy = np.stack(
        [y_model[t - window + 1: t + 1].reshape(window, 1) for t in t_index]
    ).astype(np.float32)

    y_t = y_model[t_index].astype(np.float32)             # (N,)
    y_next = y_model[target_index].astype(np.float32)     # (N,)
    r_next = (y_next - y_t).astype(np.float32)            # (N,)
    return Xy, r_next, y_t, y_next


# 5. applying splits to build train/val/test sets
def apply_split_single(Xy, y, tr_mask, va_mask, te_mask):
    """
    Apply train/val/test masks to single-input data.
    Returns: (X_tr, y_tr), (X_va, y_va), (X_te, y_te)
    """
    return (Xy[tr_mask], y[tr_mask]), (Xy[va_mask], y[va_mask]), (Xy[te_mask], y[te_mask])


def apply_split_two_now(Xy, Xc, y, tr_mask, va_mask, te_mask):
    """
    Apply masks to two-input data (Model B).
    Returns: ([Xy_tr, Xc_tr], y_tr), ([Xy_va, Xc_va], y_va), ([Xy_te, Xc_te], y_te)
    """
    Xy_tr, Xc_tr, y_tr = Xy[tr_mask], Xc[tr_mask], y[tr_mask]
    Xy_va, Xc_va, y_va = Xy[va_mask], Xc[va_mask], y[va_mask]
    Xy_te, Xc_te, y_te = Xy[te_mask], Xc[te_mask], y[te_mask]
    return ([Xy_tr, Xc_tr], y_tr), ([Xy_va, Xc_va], y_va), ([Xy_te, Xc_te], y_te)


def apply_split_two_seq(Xy, Xh, y, tr_mask, va_mask, te_mask):
    """
    Apply masks to two-sequence-input data (Model C).
    Returns: ([Xy_tr, Xh_tr], y_tr), ([Xy_va, Xh_va], y_va), ([Xy_te, Xh_te], y_te)
    """
    Xy_tr, Xh_tr, y_tr = Xy[tr_mask], Xh[tr_mask], y[tr_mask]
    Xy_va, Xh_va, y_va = Xy[va_mask], Xh[va_mask], y[va_mask]
    Xy_te, Xh_te, y_te = Xy[te_mask], Xh[te_mask], y[te_mask]
    return ([Xy_tr, Xh_tr], y_tr), ([Xy_va, Xh_va], y_va), ([Xy_te, Xh_te], y_te)


def apply_split_residual(Xy, r, y_t, y_next, tr_mask, va_mask, te_mask):
    """
    Apply masks to residual dataset.
    """
    tr = (Xy[tr_mask], r[tr_mask], y_t[tr_mask], y_next[tr_mask])
    va = (Xy[va_mask], r[va_mask], y_t[va_mask], y_next[va_mask])
    te = (Xy[te_mask], r[te_mask], y_t[te_mask], y_next[te_mask])
    return tr, va, te

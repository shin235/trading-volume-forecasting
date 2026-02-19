from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def predict_persistence(y_model, t_index):
    """
    Predict y_{t+1} as y_t (persistence).
    y_model: Target. log1p(y) if use_log1p else raw.
    """
    return y_model[t_index].astype(np.float32)


def build_gru_single(window, n_in, gru_units=128, dense_units=64, lr=1e-3, dropout=0.1):
    """
    Single-input GRU model.
    Used for Model A (y history only) and for any single sequence input setting.
    """
    inp = keras.Input(shape=(window, n_in))
    x = layers.GRU(gru_units, dropout=dropout)(inp)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out, name="GRU_A")
    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=lr), loss="mse")
    return model


def build_gru_two_branch_now(window, n_feat_now, gru_units=128, dense_units=64, lr=1e-3, dropout=0.1):
    """
    Two-branch model for Model B:
    - GRU branch for target history (y sequence)
    - Dense branch for the most recent covariates (X_{t-1})
    """

    # y-seq branch
    inp_y = keras.Input(shape=(window, 1))
    h = layers.GRU(gru_units, dropout=dropout)(inp_y)

    # X_{t-1} branch
    inp_x = keras.Input(shape=(n_feat_now,))
    z = layers.Dense(dense_units, activation="relu")(inp_x)

    merged = layers.Concatenate()([h, z])
    merged = layers.Dense(dense_units, activation="relu")(merged)
    out = layers.Dense(1)(merged)

    model = keras.Model([inp_y, inp_x], out, name="GRU_B")
    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=lr), loss="mse")
    return model


def build_gru_two_branch_seq(window_y, window_x, n_feat_x, gru_units=128, dense_units=64, lr=1e-3, dropout=0.1):
    """
    Two-sequence model for Model C:
    - GRU branch for target history (y-sequence)
    - GRU branch for covariate history up to t-1 (X-sequence)
    """

    # y-seq branch
    inp_y = keras.Input(shape=(window_y, 1))
    hy = layers.GRU(gru_units, dropout=dropout)(inp_y)

    # X-history branch (sequence up to t-1)
    inp_xh = keras.Input(shape=(window_x, n_feat_x))
    hx = layers.GRU(gru_units, dropout=dropout)(inp_xh)

    merged = layers.Concatenate()([hy, hx])
    merged = layers.Dense(dense_units, activation="relu")(merged)
    out = layers.Dense(1)(merged)

    model = keras.Model([inp_y, inp_xh], out, name="GRU_C")
    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=lr), loss="mse")
    return model


def build_gru_delta(
    window,
    gru_units=128,
    dense_units=64,
    lr=1e-3,
    dropout=0.1,
):
    inp = keras.Input(shape=(window, 1), name="y_hist")
    h = layers.GRU(gru_units, dropout=dropout, name="gru")(inp)
    h = layers.Dense(dense_units, activation="relu", name="dense")(h)
    out = layers.Dense(1, name="r_hat")(h)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
    )
    return model

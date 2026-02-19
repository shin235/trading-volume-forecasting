from tensorflow import keras

cb = [keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
), keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=12, min_delta=1e-3, restore_best_weights=True), keras.callbacks.TerminateOnNaN(),
]


def train_single(model, X_tr, y_tr, X_va, y_va, epochs=100, batch_size=32, verbose=0, shuffle=False):
    """Train a single-input model with early stopping."""
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=cb,
        shuffle=shuffle,
    )
    return history


def train_two_now(model, X_tr_in, y_tr, X_va_in, y_va, epochs=100, batch_size=32, verbose=0, shuffle=False):
    """Train a two-input model (y-seq + X_{t-1}) with early stopping."""
    history = model.fit(
        X_tr_in, y_tr,
        validation_data=(X_va_in, y_va),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=cb,
        shuffle=shuffle,
    )
    return history


def train_two_seq(model, X_tr_in, y_tr, X_va_in, y_va, epochs=100, batch_size=32, verbose=0, shuffle=False):
    """Train a two-sequence-input model (y-seq + X-history) with early stopping."""
    history = model.fit(
        X_tr_in, y_tr,
        validation_data=(X_va_in, y_va),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=cb,
        shuffle=shuffle,
    )
    return history

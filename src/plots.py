import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(history_dict, title="Title"):
    tr = history_dict.get("loss", [])
    va = history_dict.get("val_loss", [])
    plt.figure()
    plt.plot(tr, label="train")
    plt.plot(va, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_test_predictions_overlay(artifacts, use_log1p=True, title="Test block: Actual vs P vs A"):
    y_true = artifacts["A_y_true_model"]
    y_pred_A = artifacts["A_y_pred_model"]
    y_pred_P = artifacts["P_y_pred_model"]

    if use_log1p:
        y_true_o = np.expm1(y_true)
        y_A_o = np.expm1(y_pred_A)
        y_P_o = np.expm1(y_pred_P)
    else:
        y_true_o = y_true
        y_A_o = y_pred_A
        y_P_o = y_pred_P

    x = np.arange(len(y_true_o))

    plt.figure()
    plt.plot(x, y_true_o, label="Actual")
    plt.plot(x, y_P_o, label="Persistence (P)")
    plt.plot(x, y_A_o, label="GRU (A)")
    plt.xlabel("test step")
    plt.ylabel("y (original scale)")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_test_predictions_overlay_residual(artifacts, use_log1p=True, title="Title"):
    """
    artifacts must contain:
      - Res_y_true_model, Res_y_pred_model
      - P_y_pred_model
    """
    y_true = artifacts["Res_y_true_model"]
    y_pred_A = artifacts["Res_y_pred_model"]
    y_pred_P = artifacts["P_y_pred_model"]

    if use_log1p:
        y_true_o = np.expm1(y_true)
        y_A_o = np.expm1(y_pred_A)
        y_P_o = np.expm1(y_pred_P)
    else:
        y_true_o = y_true
        y_A_o = y_pred_A
        y_P_o = y_pred_P

    x = np.arange(len(y_true_o))
    plt.figure()
    plt.plot(x, y_true_o, label="Actual")
    plt.plot(x, y_P_o, label="Persistence (P)")
    plt.plot(x, y_A_o, label="Residual GRU (Res)")
    plt.xlabel("test step")
    plt.ylabel("y (original scale)")
    plt.title(title)
    plt.legend()
    plt.show()

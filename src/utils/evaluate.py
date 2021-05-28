import warnings
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

colors = cycle(
    [
        "#fafa6e",
        "#d7f171",
        "#b5e877",
        "#95dd7d",
        "#77d183",
        "#5bc489",
        "#3fb78d",
        "#23aa8f",
        "#009c8f",
        "#008d8c",
        "#007f86",
        "#0b717e",
        "#1c6373",
        "#255566",
        "#2a4858",
        "#264e64",
        "#225371",
        "#1e587e",
        "#1d5d8b",
        "#1f6298",
        "#2766a5",
        "#326ab2",
        "#406ebe",
        "#5070ca",
        "#6272d4",
        "#7574de",
        "#8975e6",
        "#9d74ed",
        "#b273f3",
        "#c871f7",
        "#de6efa",
    ]
)


def plot_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    boundaries: list = None,
    n_classes: int = None,
    lw: float = 1.0,
    labels: list = None,
):
    n_classes = y_true.shape[-1] if n_classes == None else n_classes

    plt.figure()
    plt.title(f"ROC curve{' for multiclass' if n_classes > 1 else ''}")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    if boundaries != None:
        for boundary, color in zip(boundaries, colors):
            y_true = np.where(y_true > boundary, 0, 1)
            y_pred = np.where(y_pred > boundary, 0, 1)

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, label=f"bnd {boundary}")
        plt.legend()
        plt.show()
        return

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average (area = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=lw * 2,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average (area = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=lw * 2,
    )

    for i, color in zip(range(n_classes), colors):
        lbl = f"Class {i+1}" if labels == None else labels[i]

        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            color=color,
            label=f"{lbl} (area = {roc_auc[i]:.2f})",
        )

    plt.legend(loc="upper right", bbox_to_anchor=(2.07, 1.1), ncol=2)
    plt.show()


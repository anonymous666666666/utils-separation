# utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


@dataclass
class DataBundle:
    images: any              # (n_samples, 8, 8)
    targets: any             # (n_samples,)
    flat: any                # (n_samples, 64)


# data
def load_digits_data() -> DataBundle:
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    flat = digits.images.reshape((n_samples, -1))
    return DataBundle(images=digits.images, targets=digits.target, flat=flat)


def visualize_training_samples(images, targets, n: int = 4) -> None:
    _, axes = plt.subplots(nrows=1, ncols=n, figsize=(10, 3))
    for ax, image, label in zip(axes, images[:n], targets[:n]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Training: {label}")


# model
def split_data(data, targets, test_size: float = 0.5, shuffle: bool = False):
    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=test_size, shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test


def build_classifier(gamma: float = 0.001):
    return svm.SVC(gamma=gamma)


def train(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf


def predict(clf, X_test):
    return clf.predict(X_test)


#  reporting
def visualize_predictions(X_test, predicted, n: int = 4) -> None:
    _, axes = plt.subplots(nrows=1, ncols=n, figsize=(10, 3))
    for ax, image, pred in zip(axes, X_test[:n], predicted[:n]):
        ax.set_axis_off()
        ax.imshow(image.reshape(8, 8), cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {pred}")


def print_classification_report(clf, y_test, predicted) -> None:
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )


def plot_confusion_and_return(y_test, predicted) -> Tuple[any, any]:
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return disp.figure_, disp.confusion_matrix


def rebuild_report_from_confusion_matrix(cm) -> None:
    y_true, y_pred = [], []
    for gt in range(len(cm)):
        for pr in range(len(cm)):
            y_true += [gt] * cm[gt][pr]
            y_pred += [pr] * cm[gt][pr]
    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )


# pipeline 
def run_pipeline(
    *,
    gamma: float = 0.001,
    test_size: float = 0.5,
    shuffle: bool = False,
    show_plots: bool = True,
) -> None:
    # load + preview
    bundle = load_digits_data()
    if show_plots:
        visualize_training_samples(bundle.images, bundle.targets, n=4)

    # split, train, predict
    X_train, X_test, y_train, y_test = split_data(
        bundle.flat, bundle.targets, test_size=test_size, shuffle=shuffle
    )
    clf = build_classifier(gamma=gamma)
    clf = train(clf, X_train, y_train)
    predicted = predict(clf, X_test)

    # visualize predictions
    if show_plots:
        visualize_predictions(X_test, predicted, n=4)

    # reports
    print_classification_report(clf, y_test, predicted)
    _, cm = plot_confusion_and_return(y_test, predicted)
    rebuild_report_from_confusion_matrix(cm)

    if show_plots:
        plt.show()

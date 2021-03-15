from typing import Union

import h5py
import numpy as np
from tensorflow.keras.models import Model, load_model

from utils import load_dataset


def save_predictions(model: Union[str, Model], dist: str) -> None:
    """Save predictions as h5

    Args:
        model (Union[str, Model]): model save location or model instance (keras)
        dist (str): destination for save
    """
    if type(model) == str:
        model = load_model(model)

    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)

    with hdf5.File(dist, "w") as file:
        file.create_dataset("test_predictions", data=test_predictions)
        file.create_dataset("train_predictions", data=train_predictions)

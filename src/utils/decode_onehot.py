from typing import Tuple

import numpy as np

from utils import load_dataset


def decode(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    X_train, y_train, X_test, y_test = load_dataset(dataset)

    func = lambda x: np.argmax(x, axis=0)
    decoded_train = np.array(list(map(func, X_train)))
    decoded_test = np.array(list(map(func, X_test)))

    return decoded_train, decoded_test

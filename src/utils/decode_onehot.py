from typing import Tuple

import numpy as np

from utils import load_dataset


def decode(X: Tuple[ndarray, ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    X_train, X_test = X

    get_pos = lambda x: np.argmax(x, axis=0)
    decoded_train = np.array(list(map(get_pos, X_train)))
    decoded_test = np.array(list(map(get_pos, X_test)))

    return decoded_train, decoded_test

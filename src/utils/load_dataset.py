import tarfile
from os.path import exists, join
from typing import Tuple

from h5py import File
from numpy import ndarray
from requests import get


def load_dataset(
        file: str,
        directory: str = '/tmp') -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    '''Load Dataset from storage or cloud h5 format

    Args:
        file (str): dataset name
        directory (str, optional): Location to save the dataset. Defaults to '/tmp'.

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray]: X, y train, X, y test
    '''
    fname = join(directory, file)

    # get from cloud
    if not exists(f'{fname}.tar.gz'):
        res = get(f'https://storage.gorchilov.net/datasets/{file}.tar.gz',
                  allow_redirects=True)
        open(f'{fname}.tar.gz', 'wb').write(res.content)

    # extract tar gzip
    if not exists(f'{fname}_train.h5') or not exists(f'{fname}_test.h5'):
        tar = tarfile.open(f'{fname}.tar.gz', 'r:gz')
        tar.extractall(path=directory)
        tar.close()

    train_file = File(f'{fname}_train.h5', mode='r')
    test_file = File(f'{fname}_test.h5', mode='r')

    X_train = train_file['data'][:]
    y_train = train_file['labels'][:]

    X_test = test_file['data'][:]
    y_test = test_file['labels'][:]

    return (X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_dataset('FOX')
    print(X_train)

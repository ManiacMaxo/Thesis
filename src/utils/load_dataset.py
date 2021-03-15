import tarfile
from os.path import exists, join
from typing import Tuple

from h5py import File
from numpy import ndarray
from requests import get


def load_dataset(
    file: str,
    directory: str = "/tmp",
    download: bool = True,
    url: str = None,
    labels: str = "labels",
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Load Dataset from storage or cloud h5 format

    Args:
        file (str): file name (tar gzipped, file extension not necessary)
        directory (str, optional): Location to save the dataset. Defaults to '/tmp'.
        download (bool, optional): Whether to download from repo. 
        If false, 'file' should be the path to the tar file. Defaults to 'True'.
        url (str, optional): URL of cloud storage pointing to file. Defaults to None.

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray]: X, y train, X, y test
    """
    if not file.endswith(".tar.gz"):
        file += ".tar.gz"
    filename = join(directory, file)
    url = url if url else f"https://storage.gorchilov.net/datasets/{file}"

    # get from cloud
    if not exists(filename) and download:
        res = get(url, allow_redirects=True)
        open(filename, "wb").write(res.content)

    # open tarball
    tar = tarfile.open(filename, "r:gz")

    # get filenames from tarball
    files = tar.getmembers()
    train_filename = join(directory, [i.name for i in files if "train" in i.name][0])
    test_filename = join(directory, [i.name for i in files if "test" in i.name][0])

    # extract files if not already
    if not exists(train_filename) or not exists(test_filename):
        tar.extractall(path=directory)
        tar.close()

    train_file = File(train_filename, mode="r")
    test_file = File(test_filename, mode="r")

    X_train = train_file["data"][:]
    y_train = train_file[labels][:]

    X_test = test_file["data"][:]
    y_test = test_file[labels][:]

    return (X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset("FOX")
    print(X_train)

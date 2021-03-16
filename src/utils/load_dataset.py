import tarfile
from os.path import exists, join
from typing import Tuple

from h5py import File
from numpy import ndarray
from requests import get


def load_dataset(
    file: str,
    out_dir: str = "/tmp",
    download: bool = True,
    url: str = None,
    labels: str = "labels",
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Load Dataset from storage or cloud h5 format

    Args:
        file (str): file name or file path if local (tar gzipped, file extension not necessary)
        out_dir (str, optional): Location to save the dataset (or open if local). Defaults to '/tmp'.
        download (bool, optional): Whether to download from repo. 
        If false, 'file' should be the path to the tar file. Defaults to 'True'.
        url (str, optional): URL of cloud storage pointing to file. Defaults to None.
        labels (str, optional): key of labels in hdf5 file

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray]: X, y train, X, y test
    """
    file += ".tar.gz" if not file.endswith(".tar.gz") else ""
    location = join(out_dir, file) if download else file
    url = (
        url if url else f"https://storage.gorchilov.net/datasets/{file.split('/')[-1]}"
    )

    # get from cloud
    if not exists(location) and download:
        res = get(url, allow_redirects=True)
        open(location, "wb").write(res.content)

    # open tarball
    tar = tarfile.open(location, "r:gz")

    # get filenames from tarball
    files = tar.getmembers()

    train_filename = join(
        out_dir, next(filter(lambda x: "train" in x.name, files)).name
    )
    test_filename = join(out_dir, next(filter(lambda x: "test" in x.name, files)).name)

    # extract files if not already
    if not exists(train_filename) or not exists(test_filename):
        tar.extractall(path=out_dir)

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

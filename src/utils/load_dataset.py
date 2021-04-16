import tarfile
from os.path import exists, join
from sys import stdout
from typing import Tuple

from h5py import File
from numpy import ndarray
from requests import get, head
from tqdm import tqdm


def load_dataset(
    file: str,
    out_dir: str = "/tmp",
    download: bool = True,
    url: str = None,
    labels: str = "labels",
    verbose: int = 2,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Load Dataset from storage or cloud h5 format

    Args:
        file (str): File name or file path if local (tar gzipped, file extension not necessary)
        out_dir (str, optional): Location to save the dataset (or open if local). Defaults to '/tmp'.
        download (bool, optional): Whether to download from repo.
        If false, 'file' should be the path to the tar file. Defaults to 'True'.
        url (str, optional): URL of cloud storage pointing to file. Defaults to None.
        labels (str, optional): Key of labels in hdf5 file
        verbose (int, optional): Verbosity level: 2 is most, 0 is none. Defaults to 2.

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray]: X, y train, X, y test
    """
    file += ".tar.gz" if not file.endswith(".tar.gz") else ""
    location = join(out_dir, file)
    url = (
        url if url else f"https://storage.gorchilov.net/datasets/{file.split('/')[-1]}"
    )

    # get from cloud
    if not exists(location) and download:
        res = get(url, allow_redirects=True, stream=True)

        with open(location, "wb") as f:
            if verbose == 2 and "Content-Length" in head(url).headers:
                filesize = int(head(url).headers["Content-Length"])
                with tqdm(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    total=filesize * 1024,
                    file=stdout,
                    desc=file,
                ) as progress:
                    for chunk in res.iter_content(chunk_size=1024):
                        datasize = f.write(chunk)
                        progress.update(datasize)
            else:
                f.write(res.content)
                if verbose > 0:
                    print("Finished downloading file")

    # open tarball
    tar = tarfile.open(location, "r:gz")

    # get filenames from tarball
    files = list(filter(lambda x: x.name[0] != ".", tar.getmembers()))

    train_filename = join(
        out_dir, next(filter(lambda x: "train" in x.name, files)).name,
    )
    test_filename = join(out_dir, next(filter(lambda x: "test" in x.name, files)).name)

    # extract files if not already
    if not exists(train_filename) or not exists(test_filename):
        tar.extractall(path=out_dir)
        if verbose > 0:
            print("Extracted tarball")

    tar.close()

    train_file = File(train_filename, mode="r")
    test_file = File(test_filename, mode="r")

    X_train = train_file["data"][:]
    y_train = train_file[labels][:]
    train_file.close()

    X_test = test_file["data"][:]
    y_test = test_file[labels][:]
    test_file.close()

    return (X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset("FOX_3000")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

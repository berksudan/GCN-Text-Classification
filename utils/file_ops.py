from os import makedirs
from os.path import exists
from shutil import rmtree
from typing import Any, Iterable


def create_dir(dir_path: str, overwrite: bool) -> None:
    if exists(dir_path):
        if overwrite:
            rmtree(dir_path)
            makedirs(dir_path)
        else:
            print('[WARN] directory:%r already exists, not overwritten.' % dir_path)
    else:
        makedirs(dir_path)


def write_iterable_to_file(an_iterable: Iterable[Any], file_path: str, file_mode: str = 'w'):
    with open(file_path, file_mode) as f:
        f.writelines("%s\n" % item for item in an_iterable)


def check_paths(*paths: str):
    """Check paths if they exist or not"""
    for path in paths:
        if not exists(path):
            raise FileNotFoundError('Path: {path} is not found.'.format(path=path))

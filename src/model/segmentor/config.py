import yaml
from pathlib import Path

import os


def load_config(yml_path):
    return yaml.load(
        open(yml_path, "r"), Loader=yaml.FullLoader
    )


def check_os_environ(key, use):
    if key not in os.environ:
        raise ValueError(
            f"{key} is not defined in the os variables, it is required for {use}."
        )


def dataset_dir():
    check_os_environ("DATASET", "data loading")
    return os.environ["DATASET"]
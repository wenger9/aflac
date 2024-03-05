from argparse import ArgumentParser
import os
from pathlib import Path
import requests
import yaml

from ml_product_cat_affinity.utils.dbx_utils import DiscoverHandler

base_conf_path = Path("conf")


def get_experiment_base_path() -> str:
    """Get experiment base path"""
    with open(base_conf_path/"base.yml", "r", encoding="utf-8") as file:
        yaml_dict = yaml.safe_load(file)

    return yaml_dict["experiment_base_path"]


def make_dir(path) -> requests.Response:
    """Create directory in Databricks Workspace via rest api"""
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    endpoint_path = "api/2.0/workspace/mkdirs"

    url = f"{host}/{endpoint_path}"
    headers = {"Authorization": f"Bearer {token}"}
    path_json = {"path": path}

    response = requests.post(url, headers=headers, json=path_json, timeout=5)
    return response


def make_dirs(full_branch_name: str) -> None:
    """
    Create directories for multiple regions
    """
    base_path = get_experiment_base_path()
    regions = DiscoverHandler.discover_regions(base_conf_path/"models")
    env = full_branch_name.split("/")[-1]

    for region in regions:
        path = f"{base_path}/{env}/{region}"
        print(f"Create path if not exists {path}")
        make_dir(path)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create experiment paths for the environment."
        )
    parser.add_argument("env", help="Environment name.")
    args = parser.parse_args()
    make_dirs(args.env)

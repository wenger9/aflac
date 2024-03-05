import os
from pathlib import Path
import requests

from ml_product_cat_affinity.utils.config_utils import ModelConfig


def api_request(
        method: str,
        endpoint_path: str,
        json: dict = None,
        params: dict = None
):
    """Send a request to Databricks API."""

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")

    url = f"{host}/{endpoint_path}"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.request(
        method,
        url,
        headers=headers,
        params=params,
        json=json,
        timeout=5
    )

    return response


def get_model_id(model_name: str) -> str:
    """Get model id by its name."""

    endpoint_path = "api/2.0/mlflow/databricks/registered-models/get"
    params = {"name": model_name}
    response = api_request("GET", endpoint_path, params=params)

    model_id = None
    if response.status_code == 200:
        model_id = response.json()["registered_model_databricks"]["id"]
    else:
        response.raise_for_status()

    return model_id


def create_model(model_name: str) -> str:
    """Create a model placeholder in MLFlow registry."""

    endpoint_path = "api/2.0/mlflow/registered-models/create"
    json = {"name": model_name}
    response = api_request("POST", endpoint_path, json=json)

    model_id = None
    if response.status_code == 200:
        model_id = get_model_id(model_name)
    else:
        response.raise_for_status()

    return model_id


def discover_model_paths(model_config_base_path_name: str) -> list[Path]:
    """Discover paths to model configurations."""
    path = Path(model_config_base_path_name)
    model_paths = list(path.glob("*/*.yml"))
    return model_paths


def overwrite_model_permission(model_path: str) -> None:
    """Overwrite access control list for a model.
    Create model placeholder if model doesn't exist.
    """

    model_configue: ModelConfig = ModelConfig.from_path(model_path, model_path)

    try:
        model_id = get_model_id(model_configue.registry_model_name)
    except requests.HTTPError as exc:
        if exc.response.status_code == 404:
            model_id = create_model(model_configue.registry_model_name)
        else:
            raise exc

    endpoint_path = f"api/2.0/permissions/registered-models/{model_id}"
    json = {"access_control_list": model_configue.acl}
    response = api_request("PUT", endpoint_path, json=json)

    if response.status_code == 200:
        print(response.json())
    else:
        raise requests.HTTPError(response.json())


def overwrite_models_permission(model_config_base_path_name: str) -> None:
    """Overwrite access control list for all discovered models."""

    models_path = discover_model_paths(model_config_base_path_name)
    for model_path in models_path:
        overwrite_model_permission(model_path)


if __name__ == "__main__":
    model_config_base_path = Path("conf/models")
    overwrite_models_permission(model_config_base_path)

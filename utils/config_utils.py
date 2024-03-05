from abc import ABC
from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Union, List
import yaml


def json_to_yml(yaml_name: str, json_path: str = None) -> None:
    """Convert json to yaml"""

    json_filenames = [fn for fn in os.listdir(json_path) if fn.endswith(".json")]

    yml_dict = {}
    for file_name in json_filenames:
        with open(file_name, "r", encoding="utf-8") as file:
            yml_dict[file_name[:-5]] = json.load(file)

    with open(yaml_name, "w", encoding="utf-8") as file:
        yaml.dump(yml_dict, file)


def get_conf_path() -> str:
    """Get configuration path"""
    parameter = ArgumentParser()
    parameter.add_argument("--conf-file", required=False, type=str)
    namespace = parameter.parse_known_args(sys.argv)[0]
    return namespace.conf_file


def load_config(conf_path: str) -> dict:
    """Load configuration from path"""
    with open(conf_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file.read())
    return config


def get_args(*arg_names: str) -> dict:
    """Get dict with parameters from string"""
    parameter = ArgumentParser()
    for arg in arg_names:
        parameter.add_argument(f"--{arg}", required=False, type=str)

    args_namespace = parameter.parse_known_args(sys.argv)[0]

    return vars(args_namespace)


def get_flags(*flag_names: str) -> dict:
    """Get flags specified in command line."""

    parameter = ArgumentParser()
    for flag in flag_names:
        parameter.add_argument(f"--{flag}", required=False, action="store_true")

    args_namespace = parameter.parse_known_args(sys.argv)[0]
    return vars(args_namespace)


def get_notebook_context(dbutils) -> dict:
    """Get Databricks notebook context."""

    notebook_java_ref = dbutils.notebook.entry_point.getDbutils().notebook()
    notebook_context = json.loads(notebook_java_ref.getContext().toJson())

    return notebook_context


def get_username_hash(dbutils) -> str:
    """Get last 8 characters of SHA-1 of username who runs notebook."""

    notebook_context = get_notebook_context(dbutils)
    username: str = notebook_context["tags"]["user"]
    username_hash = hashlib.sha1(username.encode()).hexdigest()[-8:]

    return username_hash


def load_configs(paths: dict, default_paths: dict) -> dict:
    """Get dict with parameters"""
    configs = {}
    for key, parameter in paths.items():
        if parameter is None:
            parameter = default_paths[key]
        configs[key] = load_config(parameter)

    return configs


class Config(ABC):
    """
    Abstract base class for configuration classes.

    Methods:
        from_path: Create a configuration object from a file path.
        from_dict: Create a configuration object from a dictionary.

    """
    @classmethod
    def from_path(cls, path: str, default_path: str):
        """
        Create a configuration object from a file path.

        If the provided path is None, the default path is used.

        Args:
            path (str): The path to the configuration file.
            default_path (str): The default path to be used if path is None.

        Returns:
            Config: The created configuration object.

        """
        if path is None:
            path = default_path

        config_dict = load_config(path)
        config = cls.from_dict(config_dict)
        return config

    @classmethod
    def from_dict(cls, config: dict):
        """
        Create a configuration object from a dictionary.

        Args:
            config (dict): The dictionary containing configuration data.

        Returns:
            Config: The created configuration object.

        """
        return cls(**config)


@dataclass
class BaseConfig(Config):
    """
    References to input tables.

    Attributes
    ----------
    project_code: str
    Acronym for the project in upper case.

    experiment_base_path: str
        Base path to MLFlow experiments in Databricks Workspace.

    storage_account: str
        Name of Azure Blob Storage where model predictions are stored.

    storage_container: str
        Name of the container in Azure Blob Storage where model predictions
    are stored.

    output_base_path: str
        Base path in Azure Blob Storage container where model predictions
    are stored.

    storage_secret_scope: str
        Secret scope with Azure Blob Storage SAS token.

    storage_secret_key: str
        Secret key of Azure Blob Storage SAS token.

    inference_output_table:
        Base name of predictions file in Azure Blob Storage.

    database_name: str
        Name of database with input tables.

    feature_table_name: str
        Name of a feature table without database name.

    targets_table_name: str
        Name of a target table without database name.

    id_col: str
        Column name with entity identifier in the feature table,
    the target table and the inference input table.
    Used as a primary key in a feature table.
    Used in predictions table.

    timestamp_col: str
        Column name with update timestamp in a feature table,
    a target table and an inference input table.
    Used as a timestamp key in a feature table.
    Timestamp keys and primary keys of the feature table
    uniquely identify the feature value for an entity at a point in time.

    external_databases: dict
        Configurations of external databases for each region.

    branch_name: str
        Current Git branch name.
    """

    project_code: str
    experiment_base_path: str
    storage_account: str
    storage_container: str
    output_base_path: str
    inference_output_table: str
    database_name: str
    storage_secret_scope: str
    storage_secret_key: str
    feature_table_name: str
    targets_table_name: str
    id_col: str
    timestamp_col: str
    external_databases: dict
    branch_name: str


@dataclass
class EnvConfig(Config):
    """Environment configuration.
    
    Attributes
    ----------
    env : name
        Name of the environment.

    sample_size : int
        Sample size of training and inference dataset.

    model_stage : str
        MLFlow stage of the model.
    """
    env: str
    sample_size: int
    model_stage: str


@dataclass
class ModelConfig(Config):
    """Model target and features."""
    region: str
    model_name: str
    registry_model_name: str
    target_col: str
    pos_target: Union[str, int]
    feature_cols: list
    cat_cols: list
    num_cols: list
    source_tables: dict
    dest_tables: dict
    targets_eval_period: int
    features_eval_period: int
    acl: List[dict]

    # pylint: disable=W0237
    @classmethod
    def from_dict(cls, model_config: dict) -> "Config":
        config = cls._extend_config(model_config)
        updated_model_config = super().from_dict(config)
        return updated_model_config

    @classmethod
    def _extend_config(cls, model_config: dict) -> dict:
        mc = model_config
        mc["registry_model_name"] = f"{mc['region']}_{mc['model_name']}"
        mc["target_col"] = mc.pop("target")
        # mc["pos_target"] = mc.pop("positive_target") #replace after
        # Debugging: Check if 'positive_target' key exists in mc
        if 'positive_target' not in mc:
            print("'positive_target' key not found in mc:", mc)
        else:
            mc["pos_target"] = mc.pop("positive_target")
        mc["feature_cols"] = list(mc["feature_types"].keys())
        mc["cat_cols"] = [f for f, t in mc["feature_types"].items() if t == "categorical"]
        mc["num_cols"] = [f for f, t in mc["feature_types"].items() if t == "numeric"]
        mc.pop("feature_types")

        return mc


class SuffixHandler:
    """Class for handling suffixes of input tables.
    """

    def __init__(
            self,
            dbutils,
            base_config: BaseConfig,
            run_as_notebook: bool,
            custom_suffix: str
    ):

        self.dbutils = dbutils
        self.base_config = base_config
        self.run_as_notebook = run_as_notebook
        self.custom_suffix = custom_suffix
        self.suffix = ""

    def handle_branch_name(self):
        """Use branch name as a suffix."""

        branch_name = self.base_config.branch_name.split("/")[-1]
        self.suffix = "_".join(branch_name.split("-"))

    def handle_run_mode(self):
        """Override suffix if script runs as notebook
        Applicable only for notebooks.
        """

        if self.run_as_notebook:
            self.suffix = get_username_hash(self.dbutils)
        elif self.custom_suffix:
            raise ValueError(
                "custom_suffix is not allowed"
                "when script runs as a Databricks job"
            )

    def handle_custom_suffix(self):
        """Override suffix if custom suffix is set explicitly"""

        if self.custom_suffix:
            self.suffix = self.custom_suffix


@dataclass
class InputConfig:
    """References to input tables for machine learning
    and data engineering pipelines.
    
    database_name : str
        Name of the database with feature table and target table.

    source_db_name : str
        Name of the database with source delta tables.

    targets_table_name : str
        Full name of target table including database name.

    feature_table_name : str
        Full name of feature table including database name.
    """

    database_name: str
    source_db_name: str
    targets_table_name: str
    feature_table_name: str

    _base_config = None
    _env_config = None
    _base_table_names = None

    @classmethod
    def from_configs(
            cls,
            base_config: BaseConfig,
            env_config: EnvConfig,
            model_config: ModelConfig
    ) -> "InputConfig":
        """
        Create an InputConfig object from individual configuration objects.

        Args:
            base_config (BaseConfig): The base configuration object.
            env_config (EnvConfig): The environment configuration object.
            model_config (ModelConfig): The model configuration object.

        Returns:
            InputConfig: The created InputConfig object.

        """
        db_name = f"{base_config.database_name}_{env_config.env}"
        region = model_config.region

        source_db_name = f"{base_config.database_name}_source_{region}"

        base_table_names = {
            "targets_table_name": base_config.targets_table_name,
            "feature_table_name": base_config.feature_table_name,
        }
        full_table_names = {
            k: f"{db_name}.{region}_{tn}"
            for k, tn in base_table_names.items()
        }

        input_configue = cls(
            db_name,
            source_db_name,
            **full_table_names
        )

        input_configue._base_config = base_config
        input_configue._env_config = env_config
        input_configue._base_table_names = base_table_names

        return input_configue

    def set_table_suffixes(self, dbutils, run_as_notebook: bool, custom_suffix: str = ""):
        """Append suffixes to input table names.
        Affects only `dev` environment.
        """

        if self._env_config.env == "dev":
            suffix_handler = SuffixHandler(dbutils,
                                           self._base_config,
                                           run_as_notebook,
                                           custom_suffix)
            suffix_handler.handle_branch_name()
            suffix_handler.handle_run_mode()
            suffix_handler.handle_custom_suffix()

            for attr in self._base_table_names:
                table_name = getattr(self, attr)
                table_name = f"{table_name}_{suffix_handler.suffix}"
                setattr(self, attr, table_name)


@dataclass
class RegionConfig:
    """Class for storing parameters of the region.

    Attributes
    ----------
    region : str
        Region name.

    feature_cols : set
        Union of features of all models defined for the region.

    cat_cols : set
        Union of categorical features of all models defined for the region.

    targets_eval_period : int
        Duration in month of the period when targets are evaluated.

    features_eval_period : int
        Duration in month of the period when features are evaluated.
    """

    region: str
    feature_cols: set
    cat_cols: set
    targets_eval_period: int
    features_eval_period: int

    @classmethod
    def from_model_config_path(cls, path: str, default_path: str):
        """
        Construct RegionConfig from path to model configuration file.

        Parameters
        ----------
        path : str
            Path to any model configuration file for the region.

        default_path : str
            Path to any model configuration file for the region.
        Used if `path` is None.
        """
        if path is None:
            path = default_path

        model_config_paths = cls.get_config_paths(path)

        equal_attrs = ["region"]
        union_attrs = ["feature_cols", "cat_cols"]
        max_attrs = ["targets_eval_period", "features_eval_period"]

        region_attrs = {}
        region_attrs.update({attr: set() for attr in union_attrs})
        region_attrs.update({attr: 0 for attr in max_attrs})

        for mcp in model_config_paths:
            model_configue = ModelConfig.from_path(mcp, mcp)
            for attr in union_attrs:
                attr_value = getattr(model_configue, attr)
                region_attrs[attr] = region_attrs[attr].union(attr_value)

            for attr in max_attrs:
                attr_value = getattr(model_configue, attr)
                region_attrs[attr] = max(attr_value, region_attrs[attr])

            for attr in equal_attrs:
                attr_value = getattr(model_configue, attr)

                if attr not in region_attrs:
                    region_attrs[attr] = attr_value

                if region_attrs[attr] != attr_value:
                    raise ValueError(
                        f"attribute `{attr}` should be the same"
                        "for all models of this region"
                    )

        config = cls(**region_attrs)
        return config

    @staticmethod
    def get_config_paths(path: str) -> str:
        """Get paths to all siblings .yml files in the 
        same directory as a file specified in path.
        """
        model_config_paths = Path(path).parent.glob("*.yml")
        return model_config_paths


@dataclass
class ExternalDBConfig:
    """Configuration of external database."""

    secret_config: dict[str, str]
    hostname: str
    database: str
    port: int

    @classmethod
    def from_dict(cls, config: dict):
        """Class constructor"""
        return cls(**config)

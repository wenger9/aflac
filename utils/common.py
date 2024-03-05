from dataclasses import dataclass
from logging import Logger
from typing import Union, List

from pyspark.sql import SparkSession
from databricks import feature_store

from ml_product_cat_affinity.utils.config_utils import BaseConfig, EnvConfig, ModelConfig
from ml_product_cat_affinity.utils.config_utils import InputConfig, RegionConfig, ExternalDBConfig
from ml_product_cat_affinity.utils.config_utils import get_args, get_notebook_context


class SingletonMeta(type):
    """
    Singleton meta implementation
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        _call__ method launch before __init__ and in charge of new entities
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass
class FeatureStoreTableConfig:
    """
    Configuration data class used to unpack parameters
    when creating or loading a Feature Store table.

    Attributes:
        database_name (str)
            Name of database to use for creating the feature table
        table_name (str)
            Name of feature table
        primary_keys (string or list)
            String or list of strings, of columns to use as the primary key(s)
            Use single column (customerID) as the
            primary key for the telco churn example.
        description (str)
            [Optional] string containing attribute description of
            feature table in the Feature Store.
            Only used when creating a Feature Store table.
    """
    database_name: str
    table_name: str
    primary_keys: Union[str, List[str]]
    description: str = None


@dataclass
class TargetsTableConfig:
    """
    Configuration data class used to unpack parameters when creating or loading targets table.

    Attributes:
        database_name (str)
            Name of database to use for creating the targets table
        table_name (str)
            Name of targets table within the database
        target_col (str)
            Name of column to use as the target column
            (in telco churn example we rename this column to 'churn')
        id_col (string or list)
            [Optional] String or list of strings, of columns to use as the primary key(s)
        dbfs_path (str)
            [Optional] DBFS path to use for the targets table
            (saving as a Delta table)
    """
    database_name: str
    table_name: str
    target_col: str
    id_col: Union[str, List[str]] = None
    dbfs_path: str = None


class JobContext(metaclass=SingletonMeta):
    """Context singleton of Databricks Job.

    Attributes
    -------
    script_args : dict
        Python script arguments as a dict:
        {argument_name: argument_value}.
    Python script arguments are used if JobContext is initialized in
    Python script.

    default_args : dict
        Default arguments as a dict:
        {argument_name: argument_value}.
    Default arguments are used if JobContext is initialized in
    Databricks Notebook.

    notebook_context: dict
    Metadata of a Databricks notebook.
    """

    def __init__(
            self,
            conf_path: str,
            env: str,
            region: str,
            model_name: str,
            spark=None
    ) -> None:

        self.spark = self._prepare_spark(spark)
        self.dbutils = self._get_dbutils(self.spark)

        self.notebook_context = get_notebook_context(self.dbutils)

        self._set_args(
            env,
            conf_path,
            region,
            model_name
        )
        self.base_config = BaseConfig.from_path(
            self.script_args["base_conf"],
            self.default_args["base_conf"]
        )
        self.env_config = EnvConfig.from_path(
            self.script_args["env_conf"],
            self.default_args["env_conf"]
        )
        self.model_config = ModelConfig.from_path(
            self.script_args["model_conf"],
            self.default_args["model_conf"]
        )
        self.input_config = InputConfig.from_configs(
            self.base_config,
            self.env_config,
            self.model_config
        )
        self.input_config.set_table_suffixes(self.dbutils, self.run_as_notebook)

        self.region_config = RegionConfig.from_model_config_path(
            self.script_args["model_conf"],
            self.default_args["model_conf"]
        )

        region = self.region_config.region
        # self.external_db_config = ExternalDBConfig.from_dict(
        #     self.base_config.external_databases[region]
        # )
        self.external_db_config = ExternalDBConfig.from_dict(
            self.base_config.external_databases.get(region, {})
        )

        self._set_spark_conf()
        self.fs = feature_store.FeatureStoreClient()

    def get_experiment_name(self):
        """
        Get the experiment name.

        Returns:
            str: The experiment name.

        """
        return f"{self.base_config.experiment_base_path}/{self.env_config.env}/" \
               f"{self.model_config.region}/{self.model_config.model_name}"

    def _get_storage_path(self):
        """
        Get the storage path.

        Returns:
            str: The storage path.

        """
        storage_path = "/".join([self.base_config.output_base_path,
                                 self.env_config.env,
                                 self.model_config.region,
                                 self.model_config.model_name
                                 ])
        return storage_path

    def get_save_location(self):
        """
        Get the save location.

        Returns:
            str: The save location.

        """
        save_location = (f"abfss://{self.base_config.storage_container}"
                         f"@{self.base_config.storage_account}"
                         f".dfs.core.windows.net/{self._get_storage_path()}/")
        return save_location

    @staticmethod
    def _prepare_spark(spark) -> SparkSession:
        if not spark:
            return SparkSession.builder.getOrCreate()
        return spark

    def _prepare_logger(self) -> Logger:
        log4j_logger = self.spark._jvm.org.apache.log4j  # noqa
        return log4j_logger.LogManager.getLogger(self.__class__.__name__)

    @staticmethod
    def _get_dbutils(spark: SparkSession):
        try:
            # pylint: disable=C0415
            from pyspark.dbutils import DBUtils  # noqa

            if "dbutils" not in locals():
                utils = DBUtils(spark)
                return utils
            return locals().get("dbutils")
        except ImportError:
            return None

    def _set_args(
            self,
            env: str,
            conf_path: str,
            region: str,
            model_name: str
    ) -> None:
        """Get Python script arguments and default arguments.

        Parameters
        ----------
        conf_path : str
            Base path to configuration files.

        region : str
            Region name.

        model_name : str
            Model name.

        Returns
        -------
        None
        """

        if env not in ["dev", "staging"]:
            raise ValueError(
                "Script can run as a notebook only in `dev`"
                "and `staging` environments"
            )

        self.default_args = {
            "base_conf": f"{conf_path}/base.yml",
            "env_conf": f"{conf_path}/environments/{env}.yml",
            "model_conf": f"{conf_path}/models/{region}/{model_name}.yml"
        }
        self.script_args = get_args("base-conf", "env-conf", "model-conf")

    def _set_spark_conf(self):
        sa_name = self.base_config.storage_account
        self.spark.conf.set(f"fs.azure.account.auth.type.{sa_name}.dfs.core.windows.net",
                            "SAS")
        self.spark.conf.set(f"fs.azure.sas.token.provider.type.{sa_name}.dfs.core.windows.net",
                            "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
        self.spark.conf.set(f"fs.azure.sas.fixed.token.{sa_name}.dfs.core.windows.net",
                            self.dbutils.secrets.get(scope=self.base_config.storage_secret_scope,
                                                     key=self.base_config.storage_secret_key))

    # def _set_spark_conf(self):
    #     # Set up OAuth configuration for Azure Blob Storage access
    #     sa_name = self.base_config.storage_account
    #     self.spark.conf.set(
    #         f"spark.hadoop.fs.azure.account.oauth.provider.type.{sa_name}.dfs.core.windows.net",
    #         "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
    #     )
    #     self.spark.conf.set(
    #         "spark.databricks.delta.preview.enabled",
    #         "true"
    #     )
    #     self.spark.conf.set(
    #         f"spark.hadoop.fs.azure.account.oauth2.client.id.{sa_name}.dfs.core.windows.net",
    #         "8770d3ee-90f9-40d6-8a52-68844a121418"
    #     )
    #     self.spark.conf.set(
    #         "spark.rpc.message.maxSize",
    #         "1024"
    #     )
    #     self.spark.conf.set(
    #         f"spark.hadoop.fs.azure.account.auth.type.{sa_name}.dfs.core.windows.net",
    #         "OAuth"
    #     )
    #     self.spark.conf.set(
    #         f"spark.hadoop.fs.azure.account.oauth2.client.endpoint.{sa_name}.dfs.core.windows.net",
    #         "https://login.microsoftonline.com/0c5638da-d686-4d6a-8df4-e0552c70cb17/oauth2/token"
    #     )

    #     # Retrieve the client secret securely from Databricks secrets
    #     client_secret = self.dbutils.secrets.get(scope="AKV-AM-EUS-DEV-DSF", key="SP-DataBrick-AM-EUS-Dev-DSF")
    #     self.spark.conf.set(
    #         f"spark.hadoop.fs.azure.account.oauth2.client.secret.{sa_name}.dfs.core.windows.net",
    #         client_secret
    #     )


    @property
    def run_as_notebook(self) -> bool:
        """Flag indicates if script run as a Databricks notebook."""

        run_as_job = "jobId" in self.notebook_context["tags"]
        return not run_as_job


@dataclass
class Prefixed:
    """
    Class with dynamic attributes with prefixed values.

    Attributes:
        prefix (str): The prefix for attribute values.
        sep (str): The separator to join the prefix and attribute value.

    """
    prefix: str
    sep: str

    def set_attrs(self, attrs: dict[str, str]) -> None:
        """
         Set dynamic attributes with prefixed values.

         Args:
             attrs (dict): A dictionary of attribute names and values.

         Raises:
             ValueError: If attribute names `prefix` or `sep` are present in `attrs`.

         """
        for key, value in attrs.items():
            if key in ["prefix", "sep"]:
                raise ValueError(
                    "Attribute names `prefix` and `sep`"
                    "are not allowed"
                )

            value = self.sep.join((self.prefix, value))
            setattr(self, key, value)


def get_db_credentials(
        dbutils,
        secret_config: dict[str, str]
) -> dict[str, str]:
    """Get credentials for a database."""

    scope = secret_config.pop("scope")
    credentials = {
        k: dbutils.secrets.get(scope=scope, key=secret_name)
        for k, secret_name in secret_config.items()
    }

    return credentials


def format_jdbc_url(
        hostname: str,
        port: int,
        database: str,
        username: str,
        password: str
) -> str:
    """Format JDBC URL."""

    url = "jdbc:sqlserver://{0}:{1};database={2};user={3};password={4}"
    url = url.format(
        hostname,
        port,
        database,
        username,
        password
    )

    return url

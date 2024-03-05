from dataclasses import dataclass
import datetime
from typing import Union, List, Callable
from dateutil.relativedelta import relativedelta


from databricks.feature_store.entities.feature_table import FeatureTable
from databricks.feature_store import FeatureStoreClient
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

from ml_product_cat_affinity.utils.config_utils import get_flags
from ml_product_cat_affinity.utils.common import JobContext


def create_and_write_feature_table(fs_client: FeatureStoreClient,
                                   data: DataFrame,
                                   feature_table_name: str,
                                   primary_keys: Union[str, List[str]],
                                   timestamp_col: Union[str, List[str]],
                                   description: str) -> FeatureTable:
    """
    Create and return a feature table with the given name and primary keys,
     writing the provided Spark DataFrame to the feature table

    Parameters
    ----------
    fs_client : FeatureStoreClient
        Databricks Feature Store client.

    data : pyspark.sql.DataFrame
        Data to create this feature table
    feature_table_name : str
        A feature table name of the form <database_name>.<table_name>,
         for example dev.user_features.
    primary_keys : Union[str, List[str]]
        The feature table's primary keys.
        If multiple columns are required, specify a list of column names, for example
        ['customer_id', 'region'].
    timestamp_col: Union[str, List[str]]
        Timestamp key for train and inference
    description : str
        Description of the feature table.
    Returns
    -------
    databricks.feature_store.entities.feature_table.FeatureTable
    """

    feature_table = fs_client.create_table(
        name=feature_table_name,
        primary_keys=primary_keys,
        timestamp_keys=timestamp_col,
        schema=data.schema,
        description=description
    )

    fs_client.write_table(df=data, name=feature_table_name, mode="overwrite")

    return feature_table


def to_datetime(timestamp: Union[datetime.date, datetime.datetime]) -> datetime.datetime:
    """
    Convert a datetime.date or datetime.datetime object to a datetime.datetime object.

    Args:
        timestamp (Union[datetime.date, datetime.datetime]): The timestamp to convert.

    Returns:
        datetime.datetime: The converted timestamp as a datetime.datetime object.

    Raises:
        TypeError: If the timestamp is not of type datetime.date or datetime.datetime.

    """
    if isinstance(timestamp, datetime.datetime):
        return timestamp

    return datetime.datetime.combine(timestamp, datetime.time.min)


def agg_col(
        spark: SparkSession,
        table: str,
        col: str,
        func: Callable
):
    """Return the value aggregated with a function for a column."""

    input_df = spark.table(table)
    agg_df = input_df.agg(func(input_df[col]).alias("agg_col"))
    agg_val = agg_df.first()["agg_col"]
    return agg_val


def get_latest(
        spark: SparkSession,
        table: str,
        col: str
) -> Union[datetime.date, datetime.datetime]:
    """Get the latest timestamp in table column."""

    latest = agg_col(spark, table, col, F.max)
    return latest


def check_feature_table_exists(
        fs_client: FeatureStoreClient,
        table_name: str
) -> bool:
    """Check if Feature Store feature table exists.
    Returns True if feature table exists in Feature Store, False if not.
    """

    try:
        fs_client.get_table(table_name)
        return True
    except ValueError:
        return False


def check_features_update_ts(
        spark: SparkSession,
        table_name: str,
        timestamp_col: str,
        target_update_ts: Union[datetime.date, datetime.datetime]
) -> bool:
    """Check if feature table contains rows that are updated before
    target table update.
    Returns True if rows exist in feature table, False if not.
    """
    earliest_feature_update_ts = agg_col(spark, table_name, timestamp_col, F.min)
    if earliest_feature_update_ts is None:
        return False
    earliest_feature_update_ts = to_datetime(earliest_feature_update_ts)
    updated_before_target = earliest_feature_update_ts <= target_update_ts
    return updated_before_target


def check_cold_start(spark, fs_client, feature_table, timestamp_col, target_update_ts) -> bool:
    """
    Checks if a cold start is required based on the feature table's existence and update timestamp.

    Args:
        spark: The SparkSession object.
        fs_client: The client for interacting with the file system.
        feature_table (str): The name of the feature table.
        timestamp_col (str): The name of the timestamp column in the feature table.
        target_update_ts: The target update timestamp for cold start comparison.

    Returns:
        bool: True if a cold start is required, False otherwise.

    """
    table_exists = check_feature_table_exists(fs_client, feature_table)
    updated_before_target = None
    if table_exists:
        updated_before_target = check_features_update_ts(
            spark,
            feature_table,
            timestamp_col,
            target_update_ts
        )

    if table_exists and updated_before_target:
        cold_start = False
    else:
        cold_start = True

    return cold_start


def query_feature_eval_period(
        spark: SparkSession,
        source_db: str,
        source_table: str,
        timestamp_col: str,
        period_start: str,
        period_end: str) -> None:
    """
    Create temporary view of source table for a period.
    Name of temporary view is specified by `source_table`.

    Parameters
    ----------
    source_db : str
        Database name with source tables.

    source_table : str
        Short source table name without database name.

    timestamp_col : str
        Column name in source table for filtering by timestamp.

    """
    spark.sql(f"""
    SELECT *
    FROM {source_db}.{source_table}
    WHERE {timestamp_col} > '{period_start}'
    AND {timestamp_col} <= '{period_end}'
    """).createOrReplaceTempView(source_table)


@dataclass
class Timeline:
    """
    Timeline of feature and target tables update.

    Attributes
    ----------
    latest_txn_ts : datetime.datetime
        Timestamp of the latest transaction in transactions table
    This timestamp is the end of targets evaluation period.

    training_end_ts : datetime.datetime
        The latest possible timestamp for which model target
    can be evaluated.

    training_start_ts : datetime.datetime
        Timestamp of the earliest transaction considered for evaluation
    of model features for training
    """

    latest_txn_ts: Union[datetime.date, datetime.datetime]
    training_end_ts: Union[datetime.date, datetime.datetime]
    training_start_ts: Union[datetime.date, datetime.datetime]

    @classmethod
    def from_table(
            cls,
            spark: SparkSession,
            table_name: str,
            timestamp_col: str,
            targets_eval_period: int,
            features_eval_period: int
    ) -> "Timeline":
        """
        Set timeline based on source table.

        Parameters
        ----------
        spark: SparkSession
            Current spark session

        table_name : str
            Full source table name including database name.

        timestamp_col : str
            Column name in source table that contains timestamps of transactions.

        targets_eval_period : int
            Duration in months of the period in which targets are evaluated.
        For example, the customer with more than one transaction in this period
        is considered as repeating.

        features_eval_period : int
            Duration in months of the period in which features are evaluated.
        """
        latest_txn_ts = get_latest(spark, table_name, timestamp_col)
        # TODO? rename for better relevance with inference case or use EvalPeriod
        training_end_ts = latest_txn_ts - relativedelta(months=targets_eval_period)
        training_start_ts = training_end_ts - relativedelta(months=features_eval_period)
        args = [
            to_datetime(ts) for ts in [
                latest_txn_ts,
                training_end_ts,
                training_start_ts
            ]
        ]
        timeline = cls(*args)
        return timeline


def get_inference_input(
        spark: SparkSession,
        feature_table_name: str,
        id_col: str,
        timestamp_col: str
) -> DataFrame:
    """
    Get inference input DataFrame from most recently
    updated rows of a feature table.

    Parameters:
    -----------

    feature_table_name: str
        Full name of feature table including database name.

    id_col: str
        Column name with entity identifier in the feature table,
    the target table and the inference input dataframe.
    Used as a primary key in a feature table.
    Used in predictions table.

    timestamp_col: str
        Column name with update timestamp in a feature table,
    a target table and an inference input table.
    Used as a timestamp key in a feature table.
    Timestamp keys and primary keys of the feature table
    uniquely identify the feature value for an entity at a point in time.

    Returns
    -------
    DataFrame
        Spark dataframe that is used as inference input.
    Inference input dataframe specifies entity identifiers and point in time
    to look up in feature table.
    This dataframe is required for FeatureStoreClient.batch_score()
    """
    update_ts = get_latest(spark, feature_table_name, timestamp_col)
    data = spark.table(feature_table_name)
    data = data.where(data[timestamp_col] == update_ts)
    data = data.select([id_col])
    data = data.withColumn(timestamp_col, F.lit(update_ts))
    return data


class DuplicatesHandler:
    """Duplicated values handler."""

    def __init__(self, id_col: str, threshold: float = 0.01):
        """
        Parameters
        ----------
        id_col: str
        Name of column where duplicates are not allowed.

        threshold: float
        Maximum ratio of duplicated rows that is allowed to drop to all rows.
        From 0 to 1.
        """
        self.id_col = id_col
        self.threshold = threshold

    def _check_duplicates(self, input_df: DataFrame) -> None:
        input_df.cache()
        count = input_df.count()
        nunique = (
            input_df
            .agg(F.countDistinct(self.id_col).alias("nunique"))
            .first()["nunique"]
        )
        duplicated_ratio = 1 - nunique / count

        if duplicated_ratio > self.threshold:
            raise ValueError(
                "Duplicates threshold is exceeded."
                f"Total rows: {count}, unique rows: {nunique}"
            )

    def apply(self, input_df: DataFrame) -> DataFrame:
        """
        Return dataframe without duplicates.
        Raise exception if duplicated rows exceed the threshold.
        """
        self._check_duplicates(input_df)

        output_df = input_df.drop_duplicates(subset=[self.id_col])
        return output_df


@dataclass
class FeatureUpdateConditions:
    """Class defines feature update conditions.

    Attributes
    ----------
    training_dataset : bool
        True if flag `training-dataset` is set in command line arguments,
    otherwise False.

    table_exists : bool
        True if feature table exists, otherwise False.

    run_as_notebook : bool
        True if script run as Databricks notebook, otherwise False.
    """

    training_dataset: bool
    table_exists: bool
    run_as_notebook: bool

    @classmethod
    def from_context(
            cls,
            job_context: JobContext
    ) -> "FeatureUpdateConditions":
        """Define conditions from script context."""

        flags = get_flags("training-dataset")

        table_exists = check_feature_table_exists(
            job_context.fs,
            job_context.input_config.feature_table_name
        )

        conditions = cls(
            flags["training_dataset"],
            table_exists,
            job_context.run_as_notebook
        )
        return conditions


class FeatureUpdate:
    """
    Class defines feature update run.

    Attributes
    ----------
    eval_period_start : Union[datetime.date, datetime.datetime]
    Start of features evaluation period.

    eval_period_end : Union[datetime.date, datetime.datetime]
    End of features evaluation period.
    """

    def __init__(self,
                 conditions: FeatureUpdateConditions,
                 timeline: Timeline,
                 features_eval_period: int
                 ):
        """
        Parameters
        ----------
        conditions : FeatureUpdateConditions
            Conditions of feature update.

        timeline : Timeline
        
        features_eval_period: int
            Duration in month of the period when features are evaluated.
        """
        self._validate_conditions(conditions)
        self.eval_period_start, self.eval_period_end = self.get_params(
            conditions,
            timeline,
            features_eval_period
        )

    @staticmethod
    def _validate_conditions(condition: FeatureUpdateConditions):
        """Raise error if script runs as a Databricks notebook
        but command line argument is set.
        """

        if condition.training_dataset and condition.run_as_notebook:
            raise RuntimeError("Incorrect script context")

    @staticmethod
    def get_params(
            condition: FeatureUpdateConditions,
            timeline: Timeline,
            features_eval_period: int
    ) -> tuple:
        """Get the start and the end of feature evaluation period.

        Parameters
        ----------
        condition : FeatureUpdateConditions
            Conditions of feature update.

        timeline : Timeline
        
        features_eval_period: int
            Duration in month of the period when features are evaluated.

        Returns
        -------
        eval_period_start, eval_period_end
            Start and end of features evaluation period.
        """

        if condition.training_dataset:
            eval_period_end = timeline.training_end_ts
        elif condition.run_as_notebook and not condition.table_exists:
            eval_period_end = timeline.training_end_ts
        else:
            eval_period_end = timeline.latest_txn_ts

        eval_period_start = eval_period_end - relativedelta(months=features_eval_period)

        return eval_period_start, eval_period_end

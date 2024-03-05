import datetime
import json
from pyspark.sql import DataFrame

from mlflow.tracking.client import MlflowClient

client = MlflowClient()


def get_commit_id(notebook_path):
    """
    Get commit_id from end of the file (last line)
    """
    notebook_path = "/dbfs" + notebook_path[5:]
    line = ""
    with open(notebook_path, "r", encoding="utf-8") as f_read:
        for line in f_read:
            pass
    commit_id = line[2:-1]
    return commit_id


def remove_prefix(input_str, separator='.'):
    """Remove all characters from the beginning of the string up to and
     including the separator to prettify csv file name"""
    parts = input_str.split(separator, 1)
    return parts[1] if len(parts) > 1 else input_str


def save_inference_to_blob_storage(
        dataframe: DataFrame,
        dbutils,
        save_location: str,
        table_name: str,
        write_dt: datetime.datetime
):
    """
    Save model inference to blob storage
    """
    dt_string = write_dt.strftime("%Y-%m-%d %H:%M:%S")
    table_name = table_name + "_" + dt_string
    csv_location = save_location + "temp.folder"
    dataframe.repartition(1).write.csv(path=csv_location, mode="overwrite", header="true", sep=",")
    file = dbutils.fs.ls(csv_location)[-1].path
    file_location = save_location + table_name + '.csv'
    dbutils.fs.cp(file, file_location)
    dbutils.fs.rm(csv_location, recurse=True)


def save_meta_to_blob_storage(
        save_location: str,
        registry_model_name: str,
        model_stage: str,
        dbutils,
        inference_input_dt: datetime.datetime,
        write_dt: datetime.datetime
):
    """
    Save code and model meta information to blob storage
    """
    model_uri = f"models:/{registry_model_name}/{model_stage}"
    model_version = client.get_latest_versions(registry_model_name, stages=[model_stage])
    inference_input_dt_string = inference_input_dt.strftime("%Y-%m-%d %H:%M:%S")

    notebook_path = (dbutils.notebook.entry_point
                     .getDbutils()
                     .notebook()
                     .getContext()
                     .notebookPath()
                     .get())

    if notebook_path.startswith("dbfs"):
        commit_id = get_commit_id(notebook_path)
    else:
        commit_id = "run_as_notebook"

    meta_param2val = {"model_uri": model_uri,
                      "model_version": model_version[0].version,
                      "notebook_path": notebook_path,
                      "inference_input_dt": inference_input_dt_string,
                      "commit_id": commit_id}

    dt_string = write_dt.strftime("%Y-%m-%d %H:%M:%S")
    meta_name = "meta_info" + "_" + dt_string + ".json"
    meta_save_location = save_location + meta_name

    dbutils.fs.put(meta_save_location, json.dumps(meta_param2val))

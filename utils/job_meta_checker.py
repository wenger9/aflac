# Databricks notebook source
# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import json

from ml_product_cat_affinity.utils.common import JobContext
from ml_product_cat_affinity.utils.pipelines.save_to_blob import get_commit_id

# COMMAND ----------

job_context = JobContext("../../conf", "dev", "NOAM", "frequency_modeling")


# COMMAND ----------


def meta_check_commit():
    """
    This function is checking that each region model has model with commit_id
    """
    notebook_path = (
        job_context.dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .notebookPath()
        .get())
    commit_id = get_commit_id(notebook_path)

    base_location = job_context.get_save_location()
    check_location = "/".join(base_location.split("/")[:-3])

    for region_path in job_context.dbutils.fs.ls(check_location):
        for model_path in job_context.dbutils.fs.ls(region_path.path):
            save_location = model_path.path
            all_files = job_context.dbutils.fs.ls(save_location)

            file_names = [save_location + x.name for x in all_files if "meta_info" in x.name]
            is_commit_exists = False
            for name in file_names:
                meta_info = job_context.dbutils.fs.head(name)
                key2val = json.loads(meta_info)
                meta_commit_id = key2val["commit_id"]

                if meta_commit_id == commit_id:
                    is_commit_exists = True
                    break

            if not is_commit_exists:
                raise ValueError(f"Can not find {commit_id} commit in {save_location}")


# COMMAND ----------

meta_check_commit()

# Databricks notebook source
from pathlib import Path

import yaml

# pylint: disable=E0401
from dbx_utils import DiscoverHandler, JobFactory

# COMMAND ----------

conf_path = Path("../../conf/")


# COMMAND ----------

# Load templates
with open(conf_path / "templates.yml", "r", encoding="utf-8") as file:
    templates = yaml.safe_load(file)

# COMMAND ----------

# Load project code
with open(conf_path / "base.yml", "r", encoding="utf-8") as file:
    base_config = yaml.safe_load(file)
    project_code = base_config['project_code']

# COMMAND ----------

# Define regions and models for which jobs will be generated
model_config_base_path = conf_path / "models"
models = DiscoverHandler.discover_models(model_config_base_path)

# COMMAND ----------

# Generate jobs from template for dev environment

job_factory = JobFactory(templates, models, model_config_base_path)

# COMMAND ----------

yaml_dict = {
    "custom": job_factory.custom,
    "environments":
        {
            "dev": {
                "workflows": job_factory.build_jobs(project_code, "dev")
            },
            "staging": {
                "workflows": job_factory.build_jobs(project_code, "staging")
            },
            "prod": {
                 "workflows": job_factory.build_jobs(project_code, "prod")
            }
        }
}

# COMMAND ----------

with open(conf_path / "deployment.yml", "w", encoding="utf-8") as file:
    yaml.dump(yaml_dict, file, sort_keys=False)

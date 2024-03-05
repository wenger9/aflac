from typing import Dict
from datetime import datetime, timedelta

from pathlib import Path
import yaml


def time_to_trigger(run_time: datetime):
    """
    Convert datetime object to cron job format
    """
    # Map the run_time to the desired format
    seconds = run_time.strftime("%S")
    minutes = run_time.strftime("%M")
    hours = run_time.strftime("%H")
    days = run_time.strftime("%d")
    months = run_time.strftime("%m")

    # return the mapped values
    return f"{seconds} {minutes} {hours} {days} {months} ?"


def get_all_job_schedule():
    """
    Set delays for databricks jobs
    """
    run_time = datetime.utcnow()

    job_name2schedule = {
        "extract": 1,
        "update": 11,
        "repeat_train": 14,
        "repeat_inference": 16,
        "one_time_train": 17,
        "one_time_inference": 19
    }

    job_name2schedule = {x: run_time + timedelta(hours=y) for x, y in job_name2schedule.items()}
    job_name2schedule = {x: time_to_trigger(y) for x, y in job_name2schedule.items()}
    return job_name2schedule


def get_updated_schedule(templates_obj: Dict, env_name, region_name):
    """
    Function for getting schedule from template and dynamically replace them
    """
    job_name2schedule = get_all_job_schedule()
    job_name2schedule = {"_".join([env_name, region_name, x]): y
                         for x, y in job_name2schedule.items()}

    for key, _ in templates_obj["schedule"].items():
        if key in job_name2schedule:
            templates_obj["schedule"][key] = job_name2schedule[key]

    return templates_obj


if __name__ == "__main__":
    CONF_PATH = Path("../../conf/")
    ENV = "STAGING"
    REGION = "NOAM"

    # Load templates
    with open(CONF_PATH / "templates.yml", "r", encoding="utf-8") as file:
        base_templates = yaml.safe_load(file)

    templates = get_updated_schedule(base_templates, ENV, REGION)

    with open(CONF_PATH / "templates.yml", "w", encoding="utf-8") as file:
        yaml.dump(templates, file, sort_keys=False)

from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import List, Callable, Union, Hashable


def safe_join(iterable: List, sep: str):
    """
    Join only strings in iterable.
    """
    return sep.join((i for i in iterable if isinstance(i, str)))


def reorder_dict(dict_: dict, keys: list) -> dict:
    """Return dictionary copy with key order defined by `keys`.

    Notes
    -----
    Improves readability of generated YAML.
    """

    undefined_order_keys = set(dict_.keys()).difference(keys)
    if undefined_order_keys:
        raise ValueError("Order for some dict keys is not defined")

    return {k: dict_[k] for k in keys if k in dict_}


def replace_all(target: Union[dict, list], source: dict) -> None:
    """
    Recursively replace values in target dictionary or list inplace
    with a value from source dictionary.

    Parameters
    ----------
    target: Union[dict, list]
        Dictionary or list with values to be replaced.

    source: dict
        Values mapping.
    For example, `source={'a': 'b'}` will replace all `a` values
    in `target` with `b`.

    Returns
    -------
    None
    """
    if isinstance(target, dict):
        keys = target.keys()
    elif isinstance(target, list):
        keys = range(len(target))
    else:
        raise TypeError('`target` should be `dict` or `list`')

    for k in keys:
        target_value = target[k]
        if isinstance(target_value, Hashable):
            if target_value in source:
                target[k] = source[target_value]

        elif isinstance(target_value, (dict, list)):
            replace_all(target_value, source)


def keys_diff(
        left: dict,
        right: dict,
        except_keys: list = None
) -> set:
    """
    Return mismatched keys in `left` and `right` dictionaries
    except specified keys.
    """

    keys_difference = set(left.keys()).symmetric_difference(right.keys())
    if except_keys is not None:
        keys_difference = keys_difference.difference(except_keys)
    return keys_difference


def remove_dependency(task: dict, task_dependency: str) -> dict:
    """Remove task dependency from a task.
    """

    task = task.copy()
    if "depends_on" in task:
        dependencies = [
            d for d in task["depends_on"]
            if d["task_key"] != task_dependency
        ]

        if dependencies:
            task["depends_on"] = dependencies
        else:
            task.pop("depends_on")

    return task


def remove_task(job: dict, task_key: str) -> None:
    """Remove task from a Databricks job inplace.
    It affects tasks dependent on removed tasked.

    Parameters
    ----------
    job: dict
    Job to be modified.

    task_key: str
    Key of the task to be removed.
    """

    tasks = []
    for task_obj in job["tasks"]:
        if task_obj["task_key"] != task_key:
            task = remove_dependency(task_obj, task_key)
            tasks.append(task)

    job["tasks"] = tasks


def get_model_attrs(model_conf_path: Path) -> tuple[str, str]:
    """
    Return region and model name from .yml path.
    """
    model_name = model_conf_path.parts[-1].split(".")[0]
    region = model_conf_path.parts[-2]
    return region, model_name


def build_env_jobs(
        env: str,
        task_builders: dict[str, Callable],
        model_config_base_path: Path,
        custom: dict
):
    """
    Builds a list of jobs based on the provided environment, task builders,
    model configuration base path, and custom data.

    Args:
        env (str): The environment for which the jobs are being built.
        task_builders (dict[str, Callable]): A dictionary mapping task names
            to corresponding task builder functions.
        model_config_base_path (Path): The base path where model configuration
            .yml files are located.
        custom (dict): Custom data to be used during job construction.

    Returns:
        List[Job]: A list of Job instances representing the built jobs.

    Raises:
        RuntimeError: If no jobs are set after processing the model
            configuration files.

    """
    # Paths to all model configuration .yml files
    model_config_paths = Path(model_config_base_path).glob("*/*.yml")
    model_config_paths = sorted(model_config_paths)
    jobs = []
    for mcp in model_config_paths:
        region, model_name = get_model_attrs(mcp)
        for name, set_tasks in task_builders.items():
            if set_tasks:
                job_builder = JobBuilder(name, env, region, model_name, custom)
                set_tasks(job_builder)
                job = job_builder.build()
                jobs.append(job)

    if not jobs:
        raise RuntimeError("Jobs are not set")

    return jobs


class JobBuilder:
    """
    Builder class for constructing jobs.

    Args:
        short_name (str): The short name of the job.
        env (str): The environment for which the job is being built.
        region (str): The region associated with the job.
        model_name (str): The model name associated with the job.
        custom (dict): Custom data to be used during job construction.

    Attributes:
        short_name (str): The short name of the job.
        custom (MappingProxyType): Immutable view of the custom data dictionary.
        env (str): The environment for which the job is being built.
        region (str): The region associated with the job.
        model_name (str): The model name associated with the job.
        name (str): The name of the job, derived from the environment, region,
            model name, and short name.
        acl: The access control list for the job.
        cluster_key (str): The key representing the job cluster.
        job_clusters (list[dict]): List of job clusters associated with the job.
        parameters (list[str]): List of parameters for the job.
        tasks (dict): Dictionary of tasks associated with the job.
        schedule: The schedule associated with the job.

    Methods:
        _set_job_params: Sets the job parameters based on the provided data.
        set_task: Sets a task for the job.
        get_tasks: Retrieves the tasks associated with the job.
        set_schedule: Sets the schedule for the job.
        build: Constructs the job dictionary.

    Raises:
        RuntimeError: If no tasks are set for the job.

    """
    def __init__(self,
                 short_name: str,
                 env: str,
                 region: str,
                 model_name: str,
                 custom: dict,
                 ) -> None:
        """
        Initialize a JobBuilder instance.

        Args:
            short_name (str): The short name of the job.
            env (str): The environment for which the job is being built.
            region (str): The region associated with the job.
            model_name (str): The model name associated with the job.
            custom (dict): Custom data to be used during job construction.

        """
        self.short_name = short_name
        # `custom` should be immutable to avoid accidental modification
        self.custom = MappingProxyType(custom)
        self.env = env
        self.region = region
        self.model_name = model_name

        self._set_job_params()

        self.tasks = {}

        self.schedule = None

    def _set_job_params(self) -> None:
        """
        Set the job parameters based on the provided data.

        """
        self.name = "_".join([
            self.env.upper(),
            self.region,
            self.model_name,
            self.short_name
        ])

        self.acl = self.custom[f"{self.env}_acl"]

        self.cluster_key = f"{self.env}-{self.short_name}-cluster"
        cluster_config_name = f"{self.env}_cluster_config"
        cluster_config = self.custom[cluster_config_name]
        # dict copy
        self.job_clusters = [{"job_cluster_key": self.cluster_key}]
        self.job_clusters[0].update(cluster_config)

        self.parameters = [
            "--base-conf", "file:fuse://conf/base.yml",
            "--env-conf", f"file:fuse://conf/environments/{self.env}.yml",
            "--model-conf", f"file:fuse://conf/models/{self.region}/{self.model_name}.yml"
        ]

    def set_task(
            self,
            task_key: str,
            script: str,
            depends_on: List[dict[str, str]] = None,
            add_params: list = None
    ) -> None:
        """
        Set a task for the job.

        Args:
            task_key (str): The key for the task.
            script (str): The script to be executed for the task.
            depends_on (List[dict[str, str]], optional):
             List of tasks on which the current task depends.
            add_params (list, optional): Additional parameters for the task.

        """
        # New instance of `parameters` to avoid anchor creation
        # and modification of initial list
        params = list(self.parameters)
        if add_params is not None:
            params.extend(add_params)

        self.tasks[task_key] = {
            "job_cluster_key": self.cluster_key,
            "spark_python_task": {
                "python_file": script,
                "parameters": params}
        }

        if depends_on is not None:
            self.tasks[task_key]["depends_on"] = depends_on

    def get_tasks(self) -> List[dict]:
        """
        Get the tasks associated with the job.

        Returns:
            List[dict]: List of dictionaries representing the tasks.

        """
        tasks = []
        for key, value in self.tasks.items():
            task = {}
            task["task_key"] = key
            task.update(value)
            tasks.append(task)

        return tasks

    def set_schedule(self, name2val):
        """
        Get the tasks associated with the job.

        Returns:
            List[dict]: List of dictionaries representing the tasks.

        """
        self.schedule = name2val

    def build(self) -> dict:
        """
        Construct the job dictionary.

        Returns:
            dict: The job dictionary.

        Raises:
            RuntimeError: If no tasks are set for the job.

        """
        if not self.tasks:
            raise RuntimeError("Job tasks are not set")

        job = {
            "name": self.name,
            "schedule": self.schedule,
            "access_control_list": self.acl,
            "job_clusters": self.job_clusters,
            "tasks": self.get_tasks()
        }

        job = {k: v for k, v in job.items() if v is not None}

        return job


class DiscoverHandler:
    """
    Handler class for discovering regions and models from model configurations.

    Methods:
        discover_regions: Discover regions from model configurations.
        discover_models: Discover available combinations of region and model name
            from model configurations.

    """
    @staticmethod
    def discover_regions(model_config_base_path: str) -> List[str]:
        """
        Discover regions from model configurations.

        Args:
            model_config_base_path (str): The base path where model configurations are located.

        Returns:
            List[str]: A list of discovered regions.

        """
        model_config_path = Path(model_config_base_path)
        regions = [p.name for p in model_config_path.iterdir() if p.is_dir()]
        return regions

    @staticmethod
    def discover_models(model_config_base_path: str) -> List[tuple[str, str]]:
        """
        Discover available combinations of a region and a model name from model configurations.

        Args:
            model_config_base_path (str): The base path where model configurations are located.

        Returns:
            List[tuple[str, str]]: A list of discovered region and model name combinations.

        """
        path = Path(model_config_base_path)
        models = [
            (p.parent.name, p.name[:-4])
            for p in path.glob("*/*.yml")
        ]

        return models

    @staticmethod
    def discover_region_models(
            model_config_base_path: str,
            region: str
    ) -> List[str]:
        """Discover model names from model configurations of a region."""

        region_config_path = Path(model_config_base_path) / region
        models = [p.name[:-4] for p in region_config_path.glob("*.yml")]

        return models


class JobHandler:
    """
    Handler class for building jobs at different levels.
    """
    @staticmethod
    def build_env_jobs(
            job_templates: list,
            project_code: str,
            env: str,
            custom: dict,
            model_config_base_path: str
    ) -> List[dict]:
        """
        Build environment level jobs for the specified environment.

        Args:
            job_templates (list): List of job templates to build jobs from.
            project_code (str): Acronym for the project in upper case.
            env (str): The environment for which the jobs are being built.
            custom (dict): Custom data to be used during job construction.
            model_config_base_path (str): The base path of the model configurations.

        Returns:
            List[dict]: List of environment level jobs.

        """
        jobs = []
        for job_template in job_templates:
            # Deep copy of `custom` is intentionally avoided
            # to preserve YAML anchors in generated file

            region_name = DiscoverHandler.discover_regions(model_config_base_path)[-1]
            model_name = DiscoverHandler.discover_region_models(model_config_base_path,
                                                                region_name)[-1]
            job = job_from_template(
                job_template=deepcopy(job_template),
                project_code=project_code,
                env=env,
                job_level="environment",
                custom=custom,
                region_name=region_name,
                model_name=model_name,
            )
            jobs.append(job)

        return jobs

    @staticmethod
    def build_region_jobs(
            job_templates: list,
            project_code: str,
            env: str,
            regions: List[str],
            custom: dict,
            model_config_base_path: str
    ) -> List[dict]:
        """
        Build region level jobs for the specified regions.

        Args:
            job_templates (list): List of job templates to build jobs from.
            env (str): The environment for which the jobs are being built.
            project_code (str): Acronym for the project in upper case.
            regions (List[str]): List of regions for which the jobs are being built.
            custom (dict): Custom data to be used during job construction.
            model_config_base_path (str): The base path of the model configurations.

        Returns:
            List[dict]: List of region level jobs.

        """
        job_templates = deepcopy(job_templates)
        # Deep copy of `custom` is intentionally avoided
        # to preserve YAML anchors in generated file
        jobs = []

        for region_name in regions:
            # TODO remove model_config_base_path argument
            # and discover_region_models after P4440-338

            model_name = DiscoverHandler.discover_region_models(model_config_base_path,
                                                                region_name)[-1]
            for job_template in job_templates:
                job = job_from_template(
                    job_template=deepcopy(job_template),
                    project_code=project_code,
                    env=env,
                    job_level="region",
                    custom=custom,
                    region_name=region_name,
                    model_name=model_name,
                )
                jobs.append(job)

        return jobs

    @staticmethod
    def build_model_jobs(
            job_templates: list,
            project_code: str,
            env: str,
            models: List[tuple[str, str]],
            custom: dict
    ) -> List[dict]:
        """
            Build model level jobs for available combinations of a region
        and a model name.

        Parameters
        ----------
        job_templates: dict
        Job templates.

        project_code: str
        Acronym for the project in upper case.

        env: str
        Environment name for a Databricks Job.

        models: List[tuple(str, str)]
        Combination of a region and a model name.

        custom: dict
        Custom build parameters

        Returns
        -------
        Databricks Job configuration for dbx

        """

        jobs = []

        for region_name, model_name in models:
            for job_template in job_templates:
                # Deep copy of `custom` is intentionally avoided
                # to preserve YAML anchors in generated file
                job = job_from_template(
                    job_template=deepcopy(job_template),
                    project_code=project_code,
                    env=env,
                    job_level="model",
                    custom=custom,
                    region_name=region_name,
                    model_name=model_name
                )
                jobs.append(job)

        return jobs


def get_job_name(
        project_code: str,
        short_job_name: str,
        env: Union[str, None] = None,
        region: Union[str, None] = None,
        model_name: Union[str, None] = None
) -> str:
    """Return full job name from short job name and suffixes
    based on context.
    Undefined suffixes with `None` values are not included
    in full job name.
    """

    name_parts = [
        project_code,
        env.upper(),
        region,
        model_name,
        short_job_name
    ]
    name = safe_join(name_parts, "_")
    return name


def get_job_clusters(
        job_short_name: str,
        env: str,
        job_clusters: Union[List[dict], None],
        custom: dict
) -> dict:
    """Return cluster configuration."""

    # Get job cluster key from template if it is specified
    if job_clusters is None:
        template_cluster_key = None
    else:
        template_cluster_key = job_clusters[0]["job_cluster_key"]

    # Get cluster config for specific environment
    name_parts = [env, template_cluster_key, "cluster_config"]
    cluster_config_name = safe_join(name_parts, "_")
    cluster_config = custom[cluster_config_name].copy()

    # Add cluster key
    cluster_key = f"{env}-{job_short_name}-cluster"
    cluster_config["job_cluster_key"] = cluster_key

    # Reorder config for better readability
    key_order = ["job_cluster_key", "new_cluster"]
    cluster_config = reorder_dict(cluster_config, key_order)

    return cluster_config


def task_from_template(
        task_template: dict,
        env: str,
        region: str,
        model_name: str,
        job_cluster_key: str,
) -> dict:
    """Return task based on template.
    Applicable for `spark_python_task`.
    """
    # TODO model_name as None after P4440-338

    task = task_template.copy()
    parameters = [
        "--base-conf", "file:fuse://conf/base.yml",
        "--env-conf", f"file:fuse://conf/environments/{env}.yml",
        "--model-conf", f"file:fuse://conf/models/{region}/{model_name}.yml"
    ]

    task["job_cluster_key"] = job_cluster_key
    spark_python_task = task["spark_python_task"]

    # Replace placeholder for region-specific scripts
    spark_python_task["python_file"] = \
        spark_python_task["python_file"].replace("{region}", region)

    # Add parameters from template if they exist
    if "parameters" in spark_python_task:
        parameters.extend(spark_python_task["parameters"])

    spark_python_task["parameters"] = parameters

    key_order = ["task_key", "job_cluster_key", "spark_python_task", "depends_on"]
    task = reorder_dict(task, key_order)

    return task


def job_from_template(
        job_template: dict,
        project_code: str,
        env: str,
        job_level: str,
        custom: dict,
        region_name: str,
        model_name: str,
) -> dict:
    """
    Construct a job dictionary from a job template.

    Args:
        job_template (dict): The job template to build the job from.
        project_code (str): Acronym for the project in upper case.
        env (str): The environment for which the job is being built.
        job_level (str): The level of the job (environment, region, model).
        custom (dict): Custom data to be used during job construction.
        region_name (str): The name of the region associated with the job.
        model_name (str): The name of the model associated with the job.

    Returns:
        dict: The constructed job dictionary.

    Raises:
        ValueError: If the job_level parameter is unrecognized.

    """
    job = job_template.copy()
    short_name = job_template["name"]

    # TODO remove job_level parameter after P4440-338
    if job_level == "environment":
        job["name"] = get_job_name(project_code, job["name"], env)
    elif job_level == "region":
        job["name"] = get_job_name(project_code, job["name"], env, region_name)
    elif job_level == "model":
        job["name"] = get_job_name(project_code, job["name"], env, region_name, model_name)
    else:
        raise ValueError(f"Unrecognized job type {job_level}")

    job["access_control_list"] = custom[f"{env}_acl"]

    job["job_clusters"] = [
        get_job_clusters(
            job_short_name=short_name,
            env=env,
            job_clusters=job.get("job_clusters"),
            custom=custom
        )]

    job["tasks"] = []
    for task_template in job_template["tasks"]:
        task = task_from_template(
            task_template=task_template,
            env=env,
            region=region_name,
            model_name=model_name,
            job_cluster_key=job["job_clusters"][0]["job_cluster_key"]
        )
        job["tasks"].append(task)

    return job


def set_schedule(job, quartz_cron_expression):
    """
    Set the schedule for a job.

    Args:
        job (dict): The job dictionary.
        quartz_cron_expression (str): The Quartz cron expression for the schedule.

    Returns:
        None

    """
    job["schedule"] = {
        "quartz_cron_expression": quartz_cron_expression,
        "timezone_id": "UTC"}


def filter_templates(
        job_templates: List[dict],
        env_jobs: List[dict],
        level_jobs: List[str]
) -> List[dict]:
    """
    Return a deep copy of job templates for specific environment
    and job level.
    """
    env_jobs_dict = {j["name"]: j for j in env_jobs}

    env_job_templates = []
    for job in job_templates:

        name = job["name"]
        available = name in env_jobs_dict and name in level_jobs
        if not available:
            continue

        job_template = deepcopy(job)
        env_job = env_jobs_dict[name]
        if "exclude_tasks" in env_job:
            for task in env_job["exclude_tasks"]:
                remove_task(job_template, task)

        env_job_templates.append(job_template)

    return env_job_templates


class JobFactory:
    """
    Class for Databricks Job generation for dbx deployment configuration.
    """

    def __init__(
            self,
            templates: dict,
            models: List[tuple[str, str]],
            model_config_base_path: str
    ):

        self.templates = templates
        self.models = models
        self.model_config_base_path = model_config_base_path
        self.custom = self.build_custom()

    def build_custom(self) -> dict:
        """Build anchors for the deployment configuration file."""

        # 2nd level anchors are used in 1st level anchors
        custom_anchors: dict = self.templates["custom_anchors"]

        # 1st level anchors are used in usual yaml content
        custom = deepcopy(self.templates["custom"])

        # replace template valuew with 2nd level anchors
        replace_all(custom, custom_anchors)

        # add 1st level anchors to initial dict
        custom_anchors.update(custom)
        return custom_anchors

    def build_jobs(self, project_code: str, env: str) -> List[dict]:
        """Build Databricks Jobs configurations for specific environment."""

        job_templates: List[dict] = self.templates["job_templates"]
        env_jobs: dict[str, list] = self.templates["env_jobs"]
        job_levels: dict[str, list] = self.templates["job_levels"]
        schedule: dict[str, str] = self.templates["schedule"]
        job_key_order: List[str] = self.templates["job_key_order"]
        custom = self.custom

        regions = list(set(i[0] for i in self.models))

        jobs = []

        env_job_templates = filter_templates(
            job_templates,
            env_jobs[env],
            job_levels["environment"]
        )
        environment_jobs = JobHandler.build_env_jobs(
            job_templates=env_job_templates,
            project_code=project_code,
            env=env,
            custom=custom,
            model_config_base_path=self.model_config_base_path
        )
        jobs.extend(environment_jobs)

        region_job_templates = filter_templates(
            job_templates,
            env_jobs[env],
            job_levels["region"]
        )
        region_jobs = JobHandler.build_region_jobs(
            job_templates=region_job_templates,
            project_code=project_code,
            env=env,
            regions=regions,
            custom=custom,
            model_config_base_path=self.model_config_base_path
        )

        jobs.extend(region_jobs)

        model_job_templates = filter_templates(
            job_templates,
            env_jobs[env],
            job_levels["model"]
        )
        model_jobs = JobHandler.build_model_jobs(
            job_templates=model_job_templates,
            project_code=project_code,
            env=env,
            models=self.models,
            custom=custom
        )

        jobs.extend(model_jobs)

        for i, j in enumerate(jobs):
            # Set schedule
            expr = schedule.get(j["name"])
            if expr is not None:
                set_schedule(j, expr)

            # Reorder job keys for better readability
            jobs[i] = reorder_dict(j, job_key_order)

        return jobs

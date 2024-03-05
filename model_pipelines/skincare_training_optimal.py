# Import necessary packages
import pandas as pd
from pyspark.sql import SparkSession
import os, shutil, tempfile
import joblib
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibrationDisplay
import mlflow
from mlflow.models import infer_signature
# import mlflow.data
from catboost import CatBoostClassifier
import yaml
import kds
import shap
import itertools
from scipy.stats import ks_2samp
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, SparkTrials
from ml_product_cat_affinity.utils.common import JobContext
from ml_product_cat_affinity.utils.logger_utils import get_logger


# Parameters  in JobContext initialization apply only for notebook runs.
# They are ignored when a script runs as a Databricks Job
job_context = JobContext("../../conf", "dev", "NOAM", "skincare_modeling")

_logger = get_logger()
_logger.info("START: Skincare Model Training Optimal | skincare_training_optimal.py")

# Retrieve thee environment from the job context
environment = job_context.env_config.env
# Retrieve the sample size required, if applicable, for the current environment
sample_size = job_context.env_config.sample_size

dataset_path = ''
cat_cols = ''
best = None
spark = SparkSession.builder.appName("ML Azure -- Training Skincare Optimal").getOrCreate()
source_table = 'db_cat_aff.master_table_noam'
# experiment_name = '/GDA/ML_Projects/ml-product-cat-affinity/dbx/dev'
experiment_name = job_context.get_experiment_name()
target = 'skincare'
snapshot_date = '2023-04-01'
# limit_sample = 2000

def prepare_dataset(df, target, fraction=1.0):
    # We have to drop targets, ids, date features and sensitive features
    cols_to_drop = ['snapshot_dt', 'elc_master_id', 'consumer_id', 'brand_id', 'first_purchase_dt', 'last_purchase_dt',
                    'store_latitude', 'store_longitude', 'store_brand_cd',
                    'ethnic_group', 'ethnicity_detail', 'gender', 'language', 'age_range', 'lang_cd',
                    'mosaic_segment', 'mosaic_group', 'exists_in_cdp',
                    'first_store_id', 'last_store_id', 'most_frequent_store', 'store_name', 'closest_store', 'most_freq_store_id',
                    'deceased_flag', 'prison_flag',
                    'target_makeup', 'target_skincare', 'target_fragrance', 'target_haircare']
    df = df.sample(frac=fraction, random_state=42).copy()
    X = df.drop(cols_to_drop, axis=1).copy()
    y = df[f'target_{target}']
    # We have to convert numerics features that are actually categorical fetures
    num_to_cat = ['email_opt_in_ind', 'phone_opt_in_ind', 'dm_opt_in_ind', 'mobile_opt_in_ind', 'mobile_contactable_ind']
                 # 'first_store_id', 'last_store_id', 'most_frequent_store']
    cat_cols = X.columns[X.dtypes=='object'].to_list()
    num_cols = list(set(X.columns)-set(cat_cols+num_to_cat))
    # We fill the missing values with Unknown and 0
    for c in cat_cols:
        X[c] = X[c].fillna('Unknown')
    for c in num_to_cat:
        X[c] = X[c].astype(str).fillna('Unknown')
    for c in num_cols:
        X[c] = X[c].astype(np.float64).fillna(0.0)
    print(f"Dataset shape: row count is {X.shape[0]}, column count is {X.shape[1]}")
    print(f"Target ratio: {round(y.mean(),6)}")
    print(f"Duplicate count: {df[['elc_master_id', 'consumer_id', 'brand_id']].duplicated().sum()}")
    return (X, y)

def save_to_dbfs(dataset):
    """
    Saves input data (tuple output of train test split) to a temporary file on DBFS to be shared among cluster nodes and returns its path.
    """
    data_filename = "dataset.joblib"
    dbfs_tmp_dir = "/dbfs/ml/tmp/cat_aff"
    os.makedirs(dbfs_tmp_dir, exist_ok=True)
    dbfs_data_dir = tempfile.mkdtemp(dir=dbfs_tmp_dir)
    dbfs_data_path = os.path.join(dbfs_data_dir, data_filename)
    joblib.dump(dataset, dbfs_data_path)
    return dbfs_data_path

def load_from_dbfs(path):
    """
    Loads saved data (a tuple of pandas dataframes).
    """
    dataset = joblib.load(path)
    return dataset

def train_and_eval(params):
    """
    Trains a Catboost classifer using training data with the input parameters and evaluates it using test data.
    """

    with mlflow.start_run(nested=True):
        dataset = load_from_dbfs(dataset_path)
        X_train, X_test, y_train, y_test = dataset
        model = CatBoostClassifier(**params,
                                   cat_features=cat_cols,
                                   random_seed=42,
                                   silent=True,
                                   use_best_model=True,  # Use the best model with test eval metric
                                   eval_metric='PRAUC',
                                   od_type="Iter",  # Overfitting detector type is nr of iterations
                                   od_wait=30  # Nr of iterations to wait for the model to warm up
                                   )
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        # Make predictions
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Log model parameters
        mlflow.log_params(params=params)

        # Log metrics
        perf = {}
        # PRAUC
        perf["prauc_train"] = metrics.average_precision_score(y_train, y_train_proba)
        perf["prauc_test"] = metrics.average_precision_score(y_test, y_test_proba)

        # ROCAUC
        perf["roc_auc_train"] = metrics.roc_auc_score(y_train, y_train_proba)
        perf["roc_auc_test"] = metrics.roc_auc_score(y_test, y_test_proba)

        # Brier
        perf["brier_score_test"] = metrics.brier_score_loss(y_test, y_test_proba)

        # Coverage ratio at 3rd decile: Ratio of the target consumers covered at 30%, when we sort them by affinity
        perf_df = pd.DataFrame({'label': y_test, 'proba': y_test_proba}).sort_values(by=['proba'], ascending=[False])
        decile_3 = perf_df.shape[0] // 10 * 3
        cov_at_decile_3 = perf_df.head(decile_3).label.sum() / perf_df.label.sum()
        perf["cov_at_decile_3"] = cov_at_decile_3

        # KS score
        ks_score = ks_2samp(perf_df.query("label==0").proba, perf_df.query("label==1").proba)[0]
        perf["ks_score"] = ks_score

        mlflow.log_metrics(perf)

        # Hyperopt tries to minimize the objective function. A higher coverage value means a better model, so we must return the negative coverage.
        loss = -cov_at_decile_3
    return {'loss': loss, 'status': STATUS_OK}

def train_model(target, iteration, params, df):
    # Prepare data
    X, y = prepare_dataset(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    cat_cols = X.columns[X.dtypes == 'object'].to_list()
    with mlflow.start_run(run_name=f'{target}_{iteration}') as run:
        # Create and train model
        model = CatBoostClassifier(**params,
                                   cat_features=cat_cols,
                                   random_seed=42, metric_period=50, eval_metric='PRAUC'
                                   )
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        # Make predictions
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Log model parameters
        mlflow.log_params(params=params)

        # Log metrics
        # PRAUC
        mlflow.log_metric("PRAUC_train", metrics.average_precision_score(y_train, y_train_proba))
        prauc_test = metrics.average_precision_score(y_test, y_test_proba)
        print(f"prauc_test: {prauc_test}")
        mlflow.log_metric("PRAUC", prauc_test)
        # ROCAUC
        mlflow.log_metric("ROCAUC_train", metrics.roc_auc_score(y_train, y_train_proba))
        roc_auc_test = metrics.roc_auc_score(y_test, y_test_proba)
        print(f"roc_auc_test: {roc_auc_test}")
        mlflow.log_metric("ROCAUC", roc_auc_test)
        # Brier
        brier_score = metrics.brier_score_loss(y_test, y_test_proba)
        print(f"brier_score_test: {brier_score}")
        mlflow.log_metric("Brier Score", brier_score)

        # Coverage ratio at 3rd decile: Ratio of the target consumers covered at 30%, when we sort them by affinity
        perf_df = pd.DataFrame({'label': y_test, 'proba': y_test_proba}).sort_values(by=['proba'], ascending=[False])
        decile_3 = perf_df.shape[0] // 10 * 3
        cov_at_decile_3 = perf_df.head(decile_3).label.sum() / perf_df.label.sum()
        print(f"Coverage_at_decile 3: {cov_at_decile_3}")
        mlflow.log_metric("Coverage_at_decile_3", cov_at_decile_3)
        # KS score
        ks_score = ks_2samp(perf_df.query("label==0").proba, perf_df.query("label==1").proba)[0]
        print(f"ks_score: {ks_score}")
        mlflow.log_metric("ks_score", ks_score)

        # Log model
        signature = infer_signature(X_test, y_test_proba)
        mlflow.catboost.log_model(cb_model=model, artifact_path='model', signature=signature)

        # # Log data
        # dataset = mlflow.data.from_pandas(df)
        # mlflow.log_input(dataset=dataset, context='training')

        # Model uri later registering
        base_uri = run.info.artifact_uri

    # Register model to ML flow to make predictions on the data later
    model_full_name = f"category_affinity_{target}_final"
    model_uri = base_uri + "/model"
    new_model_version = mlflow.register_model(model_uri, model_full_name, await_registration_for=1200)
    mlflow.end_run()
    return (X_train, X_test, y_train, y_test, model)

def main():
    global dataset_path, cat_cols

    ###
    # Prepare Data
    ###
    df = spark.sql(f"""
        select      d.*
        from        {source_table} as d
        where       substr(d.consumer_id,-1) in ('1')
                and d.snapshot_dt = '{snapshot_date}'
                and d.exists_in_cdp = 'Y'
    """).limit(sample_size).toPandas()

    mlflow.set_experiment(experiment_name)
    X, y = prepare_dataset(df, target, fraction=0.3)
    cat_cols = X.columns[X.dtypes == 'object'].to_list()

    # Create dataset
    dataset = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    # Save it to DBFS
    dataset_path = save_to_dbfs(dataset)

    ###
    # Hyperparameters
    ###
    # define the search space for the hyperparameters
    search_space = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.50),
                    'num_trees': hp.quniform('iterations', 50, 500, 1),
                    'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 10, 1),
                    'depth': hp.quniform('depth', 2, 6, 1)}
    # set parallelism
    spark_trials = SparkTrials(parallelism=4)
    # select optimization algorithm
    algo = tpe.suggest

    with mlflow.start_run(run_name=f'{target}_optimal'):
        global best
        best = fmin(
            fn=train_and_eval,
            space=search_space,
            algo=algo,
            max_evals=1,
            trials=spark_trials,
            rstate=np.random.default_rng(42)  # set seed
        )
    shutil.rmtree(dataset_path, ignore_errors=True)

    ###
    # Final Training
    ###
    params = best
    X_train, X_test, y_train, y_test, model = train_model(target=target, iteration="final", params=params, df=df)


if __name__ == "__main__":
    main()

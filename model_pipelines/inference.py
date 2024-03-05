'''
NOTE
-------
Author | ngiannuzzi@estee, rjaswal@estee
-------
Objective:
- Operationalize the Product Category Affinity model(s) inference process for incoming data (delta).
- Handle data loading, preprocessing, prediction generation, and storage.

Key Functionality:
- load new data that needs to be scored (inference input).
- utilize UDFs to generate predictions.
- manage the storage of prediction results.
'''

# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import mlflow
from datetime import datetime
from pyspark.sql.functions import struct
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
import yaml
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

from ml_product_cat_affinity.utils.common import JobContext
from ml_product_cat_affinity.utils.pipelines.save_to_blob import save_inference_to_blob_storage, save_meta_to_blob_storage
from ml_product_cat_affinity.utils.logger_utils import get_logger

# spark = SparkSession.builder.appName("ML Azure -- Product Category Affinity -- Inference").getOrCreate()
period = '2023-12-01'
master_table = 'db_cat_aff.master_table_noam'
#scoring_table = 'db_cat_aff.scoring_noam'

# # --------------------
# #  B1. Instantiate JobContext()
# #     - Parametersa in JobContext initialization apply only for notebook runs. They are ignored when a script runs as a Databricks Job
# #  B2. Instantiate get_logger()
# #     - Log runtime events
# # --------------------
job_context = JobContext("../../conf", "dev", "NOAM", "inference")
_logger = get_logger()
_logger.info("START Job: Product Category Affinity Inference | inference.py | Running...")


def get_model_details(category, period):
    # Load latest model binary
    model = mlflow.catboost.load_model(f"models:/category_affinity_{category}_final/{'latest'}")
    features = model.feature_names_ # Get feature names

    cat_idx = model.get_cat_feature_indices() # Get categorical feture indexes
    cat_features = [features[i] for i in cat_idx] # Get categorical feature names
    num_to_cat = list(set(['email_opt_in_ind', 'phone_opt_in_ind', 'dm_opt_in_ind',
                           'mobile_opt_in_ind', 'mobile_contactable_ind']).intersection(features))
    new_cat_features = list(set(cat_features)-set(num_to_cat))
    num_features = list(set(features)-set(cat_features))

    # Generate sql queries
    base_qry = f"""
    select snapshot_dt, elc_master_id, consumer_id, brand_id, target_makeup, target_skincare, target_fragrance,
    $input_vars
    from {master_table}
    where snapshot_dt = '{period}'
    ;
    """
    cat_sql_str = ["coalesce("+c+", 'Unknown') "+ c for c in new_cat_features ]
    num_sql_str = ["coalesce(cast("+c+" as double), 0) "+ c for c in num_features ]
    num_to_cat_sql_str = ["coalesce(cast("+c+" as string), 'Uknkown') "+ c for c in num_to_cat ]
    # Combine sql queries
    input_sql_str = ", ".join(cat_sql_str) +",\n"+ ", ".join(num_sql_str) +",\n"+ ", ".join(num_to_cat_sql_str)
    new_qry = base_qry.replace('$input_vars', input_sql_str)

    model_details = {'model': model, 'query': new_qry}
    return model_details

def get_scores(category, period):
    # Get model and prepare batch scoring queries
    model_details = get_model_details(category, period)
    model = model_details['model']
    query = model_details['query']
    df_score = spark.sql(query)
    # Broadcast model binary for parallel scoring
    braodcast_model = sc.broadcast(model)
    features = model.feature_names_ 

    @pandas_udf('double')
    def predict_pandas_udf(*cols):
        X = pd.concat(cols, axis=1)
        X.columns = features
        pred_proba = pd.Series(braodcast_model.value.predict_proba(X)[:, 1])
        return pred_proba

    df_prediction = (df_score
                     .withColumn(f'pred_proba_{category}', predict_pandas_udf(*features))
                     .selectExpr('snapshot_dt', 'elc_master_id', 'consumer_id', 'brand_id', f'pred_proba_{category}'))

    return df_prediction


def main():
    # Generate predictions
    skincare_score_df = get_scores('skincare', period)

    # Ensure the DataFrame is not empty before proceeding
    if isinstance(skincare_score_df, DataFrame) and not skincare_score_df.rdd.isEmpty():

        # # ---------------
        # #  F1. Write to hive_metastore as aff.skincare_score / for testing
        # # ---------------
        (skincare_score_df
            .write.mode("overwrite")
            .option("replaceWhere", f"snapshot_dt = '{period}'")
            .saveAsTable('skincare_score')
        )

        # # ---------------
        # #  G1. Save model inference to blob storage
        # #  G2. Save model metadata to blob storage / setup later
        # # ---------------
        save_location = job_context.get_save_location()
        print("save location: ", save_location)
        table_name = job_context.base_config.inference_output_table
        print("table name: ", table_name)
        write_dt = datetime.now()
        print("job context: ", job_context)    
        
        # Save to Blob Storage
        save_inference_to_blob_storage(
            skincare_score_df,
            dbutils=job_context.dbutils,
            save_location=save_location,
            table_name=table_name,
            write_dt=write_dt
            )

        # Set the MLflow experiment
        experiment_name = job_context.get_experiment_name()
        mlflow.set_experiment(experiment_name)

        # MLflow Tracking: Log the DataFrame as an artifact
        with mlflow.start_run():
            temp_path = "/dbfs/tmp/skincare_predictions.csv"
            skincare_score_df.toPandas().to_csv(temp_path, index=False)
            mlflow.log_artifact(local_path=temp_path, artifact_path="predictions")
            mlflow.end_run()
            
        _logger.info("Inference results saved successfully.")
    else:
        _logger.info("No predictions generated. DataFrame is empty.")


if __name__ == "__main__":
    main()




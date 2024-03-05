from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException

def verify_and_save_udf_check(spark: SparkSession):
    """
    Checks if UDFs are registered and functioning, and saves the list of functions to Hive.

    Args:
    spark (SparkSession): The active Spark session.
    """
    # Check if UDFs are registered and functioning
    try:
        test_df = spark.createDataFrame([(1, 2, 3, 4)], ["frequency", "recency", "T", "monetary_value"])
        test_df.createOrReplaceTempView("test_data")
        test_result = spark.sql("""
            SELECT clv(frequency, recency, T, monetary_value) as clv_value,
                   probability_alive(frequency, recency, T) as prob_alive
            FROM test_data
        """)
        test_result.show()
        print("UDFs are registered and functioning correctly.")
    except AnalysisException as e:
        print(f"Error occurred: {e}")
        return

    # Save the list of functions to Hive
    udf_check_df = spark.sql("SHOW FUNCTIONS")
    udf_check_df.write.mode("overwrite").saveAsTable("clv.udf_check")
    print("List of functions saved to Hive.")

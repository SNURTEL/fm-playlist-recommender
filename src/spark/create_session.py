from pyspark.sql import SparkSession


def create_session() -> SparkSession:
    spark = SparkSession.builder \
        .master("local[*]") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set(
        "spark.sql.execution.arrow.pyspark.fallback.enabled", "true"
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

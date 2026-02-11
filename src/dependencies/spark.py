"""
spark.py
~~~~~~~~

Module containing helper function for use with Apache Spark
"""
import json
import os
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pyspark.sql import SparkSession

from src.dependencies import logging
from src.dependencies.constant import APP_NAME

load_dotenv()
class SparkIRSystem:
    """
    IR System using Spark
    """
    def __init__(self):
        master = os.getenv('SPARK_MASTER', 'local[*]')
        is_remote = master.startswith("spark://")
        import sys
        python_path = sys.executable
        os.environ['PYSPARK_PYTHON'] = python_path
        os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

        spark_builder = (
            SparkSession
            .builder
            .master(master)
            .appName(APP_NAME)
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .config("spark.driver.maxResultSize", "8g")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.pyspark.python", python_path)
            .config("spark.pyspark.driver.python", python_path)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        )
        if is_remote:
            # 1. Tell executors to use the Python path INSIDE the Docker image
            spark_builder = spark_builder.config("spark.pyspark.python", "/usr/bin/python3")
            
            # 2. Tell the workers to connect back to your computer's IP via the Docker gateway
            # Most Linux systems use 172.17.0.1 or 172.18.0.1 for the Docker bridge
            spark_builder = spark_builder.config("spark.driver.host", "172.18.0.1") 
            
            # 3. Use 8g for driver since it's local, 4g for executors in Docker
            spark_builder = spark_builder.config("spark.driver.memory", "8g")
            spark_builder = spark_builder.config("spark.executor.memory", "4g")
        else:
            # Local mode settings
            spark_builder = spark_builder.config("spark.driver.memory", "8g")
            spark_builder = spark_builder.config("spark.executor.memory", "8g")
        self.spark_sess = spark_builder.getOrCreate()
        self.logger = logging.Log4j(self.spark_sess)
        self.context = self.spark_sess.sparkContext

    def stop_session(self):
        self.spark_sess.stop()
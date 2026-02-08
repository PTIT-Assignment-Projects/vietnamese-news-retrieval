"""
spark.py
~~~~~~~~

Module containing helper function for use with Apache Spark
"""
import json
import os
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pyspark import SparkFiles
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
        spark_builder = (
            SparkSession
            .builder
            .master(master)
            .appName(APP_NAME))
        self.spark_sess = spark_builder.getOrCreate()
        self.logger = logging.Log4j(self.spark_sess)
        self.context = self.spark_sess.sparkContext

    def stop_session(self):
        self.spark_sess.stop()

import logging
from pyspark.sql import SparkSession


class Log4j:
    """Wrapper class for Log4j JVM object.

    :param spark_session: SparkSession object.
    """

    def __init__(self, spark_session: SparkSession):
        # get spark app details with which to prefix all messages
        conf = spark_session.sparkContext.getConf()
        app_id = conf.get('spark.app.id', 'n/a')
        app_name = conf.get('spark.app.name', 'pyspark-app')

        message_prefix = f"<{app_name} {app_id}>"
        
        # Access JVM bridge using getattr to avoid PyCharm's protected member warning
        jvm = getattr(spark_session, "_jvm", None)

        if jvm:
            # Try Log4j 2 (Spark 3.3+)
            try:
                log4j = jvm.org.apache.logging.log4j
                self.logger = log4j.LogManager.getLogger(message_prefix)
                self.log_type = 'log4j2'
            except (AttributeError, TypeError):
                # Fallback to Log4j 1.x
                try:
                    log4j = jvm.org.apache.log4j
                    self.logger = log4j.LogManager.getLogger(message_prefix)
                    self.log_type = 'log4j'
                except (AttributeError, TypeError):
                    # Fallback to Python logging
                    self.logger = logging.getLogger(message_prefix)
                    self.log_type = 'python'
        else:
            # Fallback for Spark Connect or environments without JVM bridge
            self.logger = logging.getLogger(message_prefix)
            self.log_type = 'python'

    def error(self, message):
        """Log an error."""
        self.logger.error(message)
        return None

    def warn(self, message):
        """Log a warning."""
        # Log4j uses 'warn', Python logging uses 'warning'
        if self.log_type == 'python':
            self.logger.warning(message)
        else:
            self.logger.warning(message)
        return None

    def info(self, message):
        """Log information."""
        self.logger.info(message)
        return None
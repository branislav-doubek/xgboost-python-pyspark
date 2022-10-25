import logging
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


class ProjectContext():
    def __init__(self, config):
        self.config = config
        self.sc = SparkContext
        self.spark = SparkSession
        try:
            # Try to import PySpark and initiate spark session
            self.init_spark_session()
        except ImportError as e:
            logger.warning(
                "PySpark library is not installed, if you require "
                "processing Spark, please install it."
            )

    def init_spark_session(self) -> None:
        """Initialises a SparkSession using the config defined in
        project's conf folder.
        """

        app_name = 'GA Model - testing'
        spark_conf = SparkConf()
        for k, v in self.config['spark_config'].items():
            spark_conf.set(k, v)
        spark_conf.setMaster('yarn')

        spark_session_conf = (
            SparkSession.builder.appName(app_name)
                .enableHiveSupport()
                .config(conf=spark_conf)
        )

        _spark_session = spark_session_conf.getOrCreate()




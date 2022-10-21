#!/bin/sh
export SERVICE_HOME="$(cd "`dirname "$0"`"; pwd)"

# define your environment variable
export JAVA_HOME="/usr/lib/jvm/java"
export SPARK_HOME='/usr/local/spark-2.4.0-bin-hadoop2.6'

EXEC_PY=$1

spark-submit --name 'spark xgb sample' \
             --master local \
             --jars xgboost-python-pyspark/pyspark_xgb/jars/xgboost4j-spark-0.82.jar,xgboost-python-pyspark/pyspark_xgb/jars/xgboost4j-0.82.jar xgboost-python-pyspark/pyspark_xgb/${EXEC_PY}

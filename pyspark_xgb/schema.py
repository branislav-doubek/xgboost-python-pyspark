import os
import traceback
import numpy as np
from pyspark.ml.feature import VectorAssembler
from spark import get_spark, get_logger
from utils import load_config

# assert len(os.environ.get('JAVA_HOME')) != 0, 'JAVA_HOME not set'
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'

abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
PYSPARK_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
CONFIG_PATH = '/home/config.yml'
MODEL_PATH = '/output'
LOCAL_MODEL_PATH = MODEL_PATH


def main():

    try:
        config = load_config(CONFIG_PATH)
        # init spark
        spark = get_spark(app_name="pyspark-xgb")

        # get logger
        logger = get_logger(spark, "app")

        # load data
        score = spark.read.parquet(DATASET_PATH + '/score')

        # preprocess
        LABEL = 'LABEL'
        FEATURES = 'features'
        WEIGHT = 'weight'

        safe_cols = [
            'ID_CUSTOMER',
            'LABEL',
            'CD_PERIOD']
        features = [c for c in train.columns if c not in  safe_cols]

        assembler = VectorAssembler(inputCols=features, outputCol=FEATURES)

        score = assembler.transform(score).select(FEATURES, safe_cols)
        # [Optional] load model training by xgboost, predict and get validation metric
        local_model_path = LOCAL_MODEL_PATH + '/model.bin'
        xgb_cls_model = load_model(local_model_path)
        pred = predict(xgb_cls_model, score)

        pred.select(safe_cols + 'probability').write.parquet()
        slogloss = pred.withColumn('log_loss', udf_logloss(LABEL, 'probability')) \
            .agg({"log_loss": "mean"}).collect()[0]['avg(log_loss)']
        print('[xgboost] valid logloss: {}'.format(slogloss))

    except Exception:
        print(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()



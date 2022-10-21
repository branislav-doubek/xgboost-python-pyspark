import os
import traceback
import numpy as np
from pyspark.ml.feature import VectorAssembler
from spark import get_spark, get_logger
from utils import load_config, predict, load_model, 

# assert len(os.environ.get('JAVA_HOME')) != 0, 'JAVA_HOME not set'
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'

abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
PYSPARK_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
CONFIG_PATH = '/home/config.yml'
MODEL_PATH = '/output'
DATASET_PATH = '/home/dataset'
LOCAL_MODEL_PATH = MODEL_PATH


def main():

    try:
        config = load_config(CONFIG_PATH)
        # init spark
        spark = get_spark(app_name="pyspark-xgb")

        # load data
        score = spark.read.parquet(DATASET_PATH + '/score')

        # preprocess
        LABEL = 'LABEL'
        FEATURES = 'features'

        safe_cols = [
            'ID_CUSTOMER',
            'LABEL',
            'CD_PERIOD']
        features = [c for c in score.columns if c not in  safe_cols]

        assembler = VectorAssembler(inputCols=features, outputCol=FEATURES)

        score = assembler.transform(score).select(FEATURES)
        # [Optional] load model training by xgboost, predict and get validation metric
        local_model_path = LOCAL_MODEL_PATH + '/model.bin'
        xgb_cls_model = load_model(local_model_path)
        pred = predict(xgb_cls_model, score)

        pred.write.parquet("/data_output/prediction")

    except Exception:
        print(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()




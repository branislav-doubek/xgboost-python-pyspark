import os
import logging
import traceback
import numpy as np
from pyspark.ml.feature import VectorAssembler
from spark import ProjectContext
from pyspark.sql import SparkSession
from utils import create_feature_map, create_feature_imp, load_config, get_default_params
from utils import weight_mapping, train_model, load_model, predict, udf_logloss, save_model, optimize
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'

abspath = os.path.abspath(__file__)
logging.basicConfig(level=logging.INFO)

logger_params = logging.getLogger('best params log')
handler_params = logging.FileHandler('/output/best_params.log')
logger_params.addHandler(handler_params)

PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
PYSPARK_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
DATASET_PATH = '/home/dataset'
CONFIG_PATH = '/home/config.yml'
MODEL_PATH = '/output'
LOCAL_MODEL_PATH = MODEL_PATH

def main():

    try:
        config = load_config(CONFIG_PATH)
        spark = SparkSession \
        .builder \
        .appName('xgb-docker') \
        .master("local") \
        .getOrCreate()

        # load data
        train = spark.read.parquet(DATASET_PATH + '/train')
        valid = spark.read.parquet(DATASET_PATH + '/valid')
        safe_cols = config['safe_cols']
        LABEL = config['label_col']
# preprocess
        LABEL = 'LABEL'
        FEATURES = 'features'
        WEIGHT = 'weight'

        safe_cols = [
            'ID_CUSTOMER',
            'LABEL',
            'CD_PERIOD']
        #FEATURES = 'features'
        # WEIGHT = 'weight'
        features = [c for c in train.columns if c not in safe_cols]
        assembler = VectorAssembler(inputCols=features, outputCol=FEATURES)
        train, weights = weight_mapping(train, LABEL)
        valid = weight_mapping(valid, LABEL, weights)[0]
        train = assembler.transform(train).select(FEATURES, LABEL, WEIGHT)
        valid = assembler.transform(valid).select(FEATURES, LABEL, WEIGHT)
        if config['mode'] == 'hyperopt' and 'search_space' in config.keys():
            best_params = optimize(train, valid, FEATURES, LABEL, WEIGHT, config, spark)
        else:
            best_params = get_default_params(config)
        logger_params.info('Best parameters: %s', best_params)
        jmodel = train_model(train, best_params, FEATURES, LABEL, WEIGHT, spark)
        model_path = MODEL_PATH + '/model.bin'
        save_model(jmodel, model_path)

        # get feature score
        imp_type = "gain"
        feature_map_path = MODEL_PATH + '/feature.map'
        create_feature_map(feature_map_path, features)
        jfeatureMap = jmodel.nativeBooster().getScore(feature_map_path, imp_type)
        f_imp = dict()
        for feature in features:
            if not jfeatureMap.get(feature).isEmpty():
                f_imp[feature] = jfeatureMap.get(feature).get()
        feature_imp_path = MODEL_PATH + '/feature.imp'
        create_feature_imp(feature_imp_path, f_imp)

        # [Optional] load model training by xgboost, predict and get validation metric
        local_model_path = LOCAL_MODEL_PATH + '/model.bin'
        xgb_cls_model = load_model(local_model_path, spark)
        pred = predict(xgb_cls_model, valid, spark)
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




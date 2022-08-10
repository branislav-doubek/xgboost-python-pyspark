import os
import shutil
import traceback
import numpy as np

import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.wrapper import JavaWrapper

from spark import get_spark, get_logger
from schema import get_btrain_schema
from utils import create_feature_map, create_feature_imp, print_summary
from models_utils import (
    DevXGBoostModelHyperparams,
    DevXGBoostModel)
import optuna
# assert len(os.environ.get('JAVA_HOME')) != 0, 'JAVA_HOME not set'
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'

abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
PYSPARK_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
DATASET_PATH = '/home/dataset'
MODEL_PATH = PYSPARK_PROJ_PATH + '/binary_model'
LOCAL_MODEL_PATH = PARENT_PROJ_PATH + '/python_xgb/binary_model'



def main():

    try:

        # init spark
        spark = get_spark(app_name="pyspark-xgb")

        # get logger
        logger = get_logger(spark, "app")

        # load data
        train = spark.read.parquet(DATASET_PATH + '/train')
        valid = spark.read.parquet(DATASET_PATH + '/valid')
        test = spark.read.parquet(DATASET_PATH + '/test')

        safe_cols = [
            'ID_CUSTOMER',
            'label',
            'CD_PERIOD']

        feature_cols = [col for col in train.columns if col not in safe_cols]
        label_col = 'label'

        def objective(trial):
            max_depth = trial.suggest_int('max_depth', 5, 30)
            eta = trial.suggest_loguniform('eta', 0.001, 0.01)
            gamma = trial.suggest_float('gamma', 1, 30)
            subsample = trial.suggest_float('subsample', 0.01, 0.6)
            min_child_weight = trial.suggest_float('min_child_weight', 1, 50)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.3, 1)
            params = DevXGBoostModelHyperparams()
            params.eval_metric = 'aucpr'
            params.objective = 'binary:logistic'
            params.max_depth = max_depth
            params.eta = eta
            params.gamma = gamma
            params.subsample = subsample
            params.min_child_weight = min_child_weight
            params.colsample_bytree = colsample_bytree
            logging.info('parameters')
            logging.info(params)
            model1 = DevXGBoostModel(params, XGBoostEstimator, feature_cols, label_col, actions=[])
            model2 = DevXGBoostModel(params, XGBoostEstimator, feature_cols, label_col, actions=[])
            train1 = train.filter(f.col('CD_PERIOD') < 202203)
            train2 = train.filter(f.col('CD_PERIOD') >= 202203)
            valid1 = valid.filter(f.col('CD_PERIOD') < 202203)
            valid2 = valid.filter(f.col('CD_PERIOD') >= 202203)
            score1 = model1.cross_validate(train1, valid1)
            score2 = model2.cross_validate(train2, valid2)
            score = (score1+score2) / 2
            return score
    
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=2)

        params = DevXGBoostModelHyperparams()
        params.colsample_bytree = best_params['colsample_bytree']
        params.eta = best_params['eta']
        params.gamma = best_params['gamma']
        params.max_depth = best_params['max_depth']
        params.min_child_weight = best_params['min_child_weight']
        params.subsample = best_params['subsample']

        print('Best parameters of the model:')
        print(params) 

        model = DevXGBoostModel(params, XGBoostEstimator, feature_cols, label_col, actions=[])
        score = model.cross_validate(train, valid)


        from pyspark.mllib.evaluation import MulticlassMetrics
        predictions = model.predict(test)
        predictions_labels = predictions.rdd.map(lambda x: (x['prediction'], x['label']))
        metrics = MulticlassMetrics(predictions_labels)

    except Exception:
        logger.error(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()

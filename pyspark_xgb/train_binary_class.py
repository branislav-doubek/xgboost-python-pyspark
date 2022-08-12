import os
import shutil
import traceback
import numpy as np

import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.wrapper import JavaWrapper
from itertools import chain
from pyspark.ml.feature import VectorAssembler, QuantileDiscretizer
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics

from spark import get_spark, get_logger
from schema import get_btrain_schema
from utils import create_feature_map, create_feature_imp, print_summary

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


def udf_logloss(truth, pred, eps=1e-15):
    import math

    def logloss_(truth, pred):
        pred = eps if float(pred[1]) < eps else float(pred[1])
        return -(truth * math.log(pred) + (1 - truth) * math.log(1 - pred))
    return F.udf(logloss_, FloatType())(truth, pred)

def calculate_weights(label_col, label):
    y_collect = label_col.groupBy(label).count().collect()
    unique_y = [x[label] for x in y_collect]
    total_y = sum([x['count'] for x in y_collect])
    unique_y_count = len(y_collect)
    bin_count = [x['count'] for x in y_collect]
    weights = {i: ii for i, ii in zip(unique_y, total_y / (
                unique_y_count * np.array(bin_count)))}
    return weights

def weight_mapping(df: DataFrame, label, weights=False):
    if not weights:
        weights = calculate_weights(df.select(label), label)
    mapping_expr = F.create_map([F.lit(x) for x in chain(*weights.items())])
    return df.withColumn('weight', mapping_expr.getItem(F.col(label))), weights

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
        train, weights = weight_mapping(train, LABEL)
        valid = weight_mapping(valid, LABEL, weights)[0]

        train = assembler.transform(train).select(FEATURES, LABEL, WEIGHT)
        valid = assembler.transform(valid).select(FEATURES, LABEL, WEIGHT)
        
        # set param map
        xgb_params = {
            "eta": 0.1, "eval_metric": "aucpr",
            "gamma": 0, "max_depth": 5, "min_child_weight": 1.0,
            "objective": "binary:logistic", "seed": 0,
            # xgboost4j only
            "num_round": 100, "num_early_stopping_rounds": 10,
            "maximize_evaluation_metrics": False,   # minimize logloss
            "num_workers": 1, "use_external_memory": False,
            "missing": np.nan,
        }
        scala_map = spark._jvm.PythonUtils.toScalaMap(xgb_params)

        # set evaluation set
        eval_set = {'eval': valid._jdf}
        scala_eval_set = spark._jvm.PythonUtils.toScalaMap(eval_set)

        logger.info('training')
        j = JavaWrapper._new_java_obj(
            "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier", scala_map) \
            .setFeaturesCol(FEATURES).setLabelCol(LABEL).setWeightCol(WEIGHT) \
            .setEvalSets(scala_eval_set)
        jmodel = j.fit(train._jdf)
        print_summary(jmodel)

        # get validation metric
        preds = jmodel.transform(valid._jdf)
        pred = DataFrame(preds, spark)
        print(pred.show())
        predictions_labels = pred.rdd.map(
                    lambda x: (x['prediction'], x['LABEL']))
        metrics = MulticlassMetrics(predictions_labels)
        logger.info('[xgboost4j] valid logloss: {}'.format(slogloss))

        # save model - using native booster for single node library to read
        model_path = MODEL_PATH + '/model.bin'
        logger.info('save model to {}'.format(model_path))
        jbooster = jmodel.nativeBooster()
        jbooster.saveModel(model_path)

        # get feature score
        imp_type = "gain"
        feature_map_path = MODEL_PATH + '/feature.map'
        create_feature_map(feature_map_path, features)
        jfeatureMap = jbooster.getScore(feature_map_path, imp_type)
        f_imp = dict()
        for feature in features:
            if not jfeatureMap.get(feature).isEmpty():
                f_imp[feature] = jfeatureMap.get(feature).get()
        feature_imp_path = MODEL_PATH + '/feature.imp'
        create_feature_imp(feature_imp_path, f_imp)

        # [Optional] load model training by xgboost, predict and get validation metric
        local_model_path = LOCAL_MODEL_PATH + '/model.bin'
        if os.path.exists(local_model_path):
            logger.info('load model from {}'.format(local_model_path))
            scala_xgb = spark.sparkContext._jvm.ml.dmlc.xgboost4j.scala.XGBoost
            jbooster = scala_xgb.loadModel(local_model_path)

            # uid, num_class, booster
            xgb_cls_model = JavaWrapper._new_java_obj(
                "ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel",
                "xgbc", 2, jbooster)

            jpred = xgb_cls_model.transform(test._jdf)
            pred = DataFrame(jpred, spark)
            slogloss = pred.withColumn('log_loss', udf_logloss(LABEL, 'probability')) \
                .agg({"log_loss": "mean"}).collect()[0]['avg(log_loss)']
            logger.info('[xgboost] valid logloss: {}'.format(slogloss))
        else:
            logger.info(
                "local model is not exist, call python_xgb/train_binary.py to get the model "
                "and compare logloss between xgboost and xgboost4j"
            )

    except Exception:
        logger.error(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()

from typing import Optional
from spark import get_spark, get_logger
import yaml

# import joblib
import pyspark.sql.functions as f
import numpy as np
from dataclasses import dataclass
from itertools import chain
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.wrapper import JavaWrapper
import optuna
from typing import Optional
from spark import get_spark
import os

def create_feature_map(fname, features):
    '''Write feature name for xgboost to map 'fn' -> feature name
        Args:
            fname(string): file name
            features(list): feature list
    '''
    with open(fname, 'w') as f:
        for i, feature in enumerate(features):
            f.write('{0}\t{1}\tq\n'.format(i, feature))


def create_feature_imp(fname, f_imp):
    '''Write feature importance file, and sort desc based on importance
        Args:
            fname(string): file name
            f_imp(dict): {feature_name(string): importance(numeric)}
    '''
    with open(fname, 'w') as f:
        for feature, imp in sorted(f_imp.items(), key=lambda v: v[1], reverse=True):
            f.write('{:20} {:.10f}\n'.format(feature, imp))


def print_summary(jmodel):
    '''Print train and valid summary for model
        Args:
            jmodel(ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier)
    '''
    # get spark and logger
    spark = get_spark(app_name="pyspark-xgb")
    logger = get_logger(spark, "app")

    train_summary = jmodel.summary().trainObjectiveHistory()
    valid_summary = jmodel.summary().validationObjectiveHistory()
    dataset_summary = [train_summary]
    dataset_name = ['train']
    for idx in range(valid_summary.size()):
        eval_name = valid_summary.apply(idx)._1()
        eval_summary = valid_summary.apply(idx)._2()
        dataset_name.append(eval_name)
        dataset_summary.append(eval_summary)

    stop_flg = False
    for round_idx, row in enumerate(zip(*dataset_summary), 1):
        printString = "{:6} ".format('[{}]'.format(round_idx))
        for idx, r in enumerate(row):
            if r == 0:
                stop_flg = True
                break
            printString += "{:5}\t{:10}\t".format(dataset_name[idx], r)

        if stop_flg is True:
            break
        logger.info(printString)


def load_config(path):
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded

def udf_logloss(truth, pred, eps=1e-15):
    import math

    def logloss_(truth, pred):
        pred = eps if float(pred[1]) < eps else float(pred[1])
        return -(truth * math.log(pred) + (1 - truth) * math.log(1 - pred))
    return F.udf(logloss_, T.FloatType())(truth, pred)

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


def train_model(train, params, feature_col, label_col, weight_col):
    spark = get_spark(app_name="pyspark-xgb")
    scala_map = spark._jvm.PythonUtils.toScalaMap(params)
    j = JavaWrapper._new_java_obj(
        "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier", scala_map) \
        .setFeaturesCol(feature_col).setLabelCol(label_col).setWeightCol(weight_col)
    jmodel = j.fit(train._jdf)
    return jmodel

def predict(model, data):
    spark = get_spark(app_name="pyspark-xgb")
    preds = model.transform(data._jdf)
    prediction = DataFrame(preds, spark)
    return prediction

def save_model(model, path):
    print(path)
    jbooster = model.nativeBooster()
    jbooster.saveModel(path)

def load_model(path):
    spark = get_spark(app_name="pyspark-xgb")
    print(path)
    if os.path.exists(path):
        scala_xgb = spark.sparkContext._jvm.ml.dmlc.xgboost4j.scala.XGBoost
        jbooster = scala_xgb.loadModel(path)
        # uid, num_class, booster
        xgb_cls_model = JavaWrapper._new_java_obj(
            "ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel",
            "xgbc", 2, jbooster)
        return xgb_cls_model
    else:
        print('model does not exist')

def calculate_statistics(predictions, label_col, multiclass=False):
    predictions_labels = predictions.rdd.map(
                lambda x: (x['prediction'], x['class']))
    metrics = MulticlassMetrics(predictions_labels)
    labels = predictions.rdd.map(lambda lp: float(lp.class)).distinct().collect()
    score = 0
    for label in sorted(labels[1:]):
        print(
            'Class %s precision = %s' % (label, metrics.precision(float(label))))
        print(
            'Class %s recall = %s' % (label, metrics.recall(float(label))))
        print('Class %s F1 Measure = %s' % (
        label, metrics.fMeasure(label, beta=1.0)))
        if multiclass:
            score += metrics.fMeasure(label, beta=1.0)
        else:
            score += metrics.recall(label)

    if multiclass:
        score = score/(len(labels)-1)
        print('Weighted F1-Score')
    else:
        print('Recall')
    print(score)
    cm = metrics.confusionMatrix()
    print('Confusion Matrix')
    print(cm)
    return score


def cross_validate(train, valid, xgb_params, features_col, label_col, weight_col, multiclass=False, summary=False):
    # set param map
    spark = get_spark(app_name="pyspark-xgb")
    scala_map = spark._jvm.PythonUtils.toScalaMap(xgb_params)

    # set evaluation set
    eval_set = {'eval': valid._jdf}
    scala_eval_set = spark._jvm.PythonUtils.toScalaMap(eval_set)

    jmodel = train_model(train, xgb_params, features_col, label_col, weight_col)
    if summary:
        print_summary(jmodel)

    # get validation metric
    preds = predict(jmodel, valid)
    preds = preds.withColumn(label_col, F.col(label_col).cast(T.DoubleType()))
    if int(xgb_params['num_class']) > 2:
        score = calculate_statistics(preds, multiclass=True)
    else:
        score = calculate_statistics(preds, multiclass=False)
    return score


def optimize(train, valid, features_col, label_col, weight_col, cfg):
    def objective(trial):
        if cfg['max_depth']:
            max_depth = trial.suggest_int('max_depth', 
                                          cfg['max_depth'][0], 
                                          cfg['max_depth'][1])
        else:
            max_depth = trial.suggest_int('max_depth', 5, 30)
        if cfg['eta']:
            eta = trial.suggest_loguniform('eta', 
                                    cfg['eta'][0], 
                                    cfg['eta'][1])
        else:
            eta = trial.suggest_loguniform('eta', 0.001, 0.01)
        if cfg['gamma']:
            gamma = trial.suggest_float('gamma', 
                                    cfg['gamma'][0], 
                                    cfg['gamma'][1])
        else:
            gamma = trial.suggest_float('eta', 1, 30)
        if cfg['subsample']:
            subsample = trial.suggest_float('subsample', 
                                    cfg['subsample'][0], 
                                    cfg['subsample'][1])
        else:
            subsample = trial.suggest_float('subsample', 0.01, 0.6)
        if cfg['min_child_weight']:
            min_child_weight = trial.suggest_float('min_child_weight', 
                                    cfg['min_child_weight'][0], 
                                    cfg['min_child_weight'][1])
        else:
            min_child_weight = trial.suggest_float('subsample', 1, 50)
        if cfg['colsample_bytree']:
            colsample_bytree = trial.suggest_float('colsample_bytree', 
                                    cfg['colsample_bytree'][0], 
                                    cfg['colsample_bytree'][1])
        else:
            colsample_bytree = trial.suggest_float('subsample', 0.3, 1)
        
        xgb_params = {
            "eta": 0.1, "eval_metric": cfg['eval_metric'],
            "gamma": 1, "max_depth": 5, "min_child_weight": 1.0,
            "objective": cfg['objective'], "seed": 0,
            # xgboost4j only
            "num_round": 100, "num_early_stopping_rounds": 10,
            "maximize_evaluation_metrics": False,   # minimize logloss
            "num_workers": 1, "use_external_memory": False,
            "missing": np.nan,
            "num_class": cfg['num_class']
        }
        xgb_params['max_depth'] = max_depth
        xgb_params['eta'] = eta
        xgb_params['gamma'] = gamma
        xgb_params['subsample'] = subsample
        xgb_params['min_child_weight'] = min_child_weight
        xgb_params['colsample_bytree'] = colsample_bytree
        score = cross_validate(train, valid, xgb_params, features_col, label_col, weight_col, summary=False)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=cfg['n_trials'])
    
    best_params = study.best_params
    print(f'Best params {best_params}')
    return best_params


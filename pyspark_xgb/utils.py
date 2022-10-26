import yaml
import logging
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
import os

logging.basicConfig(level=logging.INFO)

logger_hyperparam = logging.getLogger('hyperparam logging')
handler_hyperparam = logging.FileHandler('/output/hyperparam.log')
logger_hyperparam.addHandler(handler_hyperparam)

logger_statistics = logging.getLogger('statistics logging')
handler_statistics = logging.FileHandler('/output/statistics.log')
logger_statistics.addHandler(handler_statistics)


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


def train_model(train, params, feature_col, label_col, weight_col, spark):
    scala_map = spark._jvm.PythonUtils.toScalaMap(params)
    j = JavaWrapper._new_java_obj(
        "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier", scala_map) \
        .setFeaturesCol(feature_col).setLabelCol(label_col).setWeightCol(weight_col)
    jmodel = j.fit(train._jdf)
    return jmodel

def predict(model, data, spark):
    preds = model.transform(data._jdf)
    prediction = DataFrame(preds, spark)
    return prediction

def save_model(model, path):
    print(path)
    jbooster = model.nativeBooster()
    jbooster.saveModel(path)

def load_model(path, spark):
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

def calculate_statistics(predictions, label_col, multiclass=False, log=False):
    predictions_labels = predictions.rdd.map(
                lambda x: (x['prediction'], x[label_col]))
    metrics = MulticlassMetrics(predictions_labels)
    predictions = predictions.withColumnRenamed(label_col, 'LABEL')
    labels = predictions.rdd.map(lambda lp: float(lp.LABEL)).distinct().collect()
    score = 0
    if multiclass:
        part_ds = labels
    else:
        part_ds = labels[1:]
    for label in sorted(part_ds):
        if log:
            logger_statistics.info('Class %s precision = %s' % (label, metrics.precision(float(label))))
            logger_statistics.info('Class %s recall = %s' % (label, metrics.recall(float(label))))
            logger_statistics.info('Class %s F1 Measure = %s' % (label, metrics.fMeasure(label, beta=1.0)))
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
        if log:
            logger_statistics.info('Weighted F1-Score: %s', score)
        print('Weighted F1-Score')
    else:
        print('Recall')
        if log:
            logger_statistics.info('Recall: %s', score)
    print(score)
    cm = metrics.confusionMatrix()
    print('Confusion Matrix')
    if log:
        logger_statistics.info('CM: %s', cm)
    print(cm)
    return score


def cross_validate(train, valid, xgb_params, features_col, label_col, weight_col, spark, multiclass=False, summary=False, log=False):
    # set param map
    jmodel = train_model(train, xgb_params, features_col, label_col, weight_col, spark)
    if summary:
        print_summary(jmodel)

    # get validation metric
    preds = predict(jmodel, valid, spark)
    preds = preds.withColumn(label_col, F.col(label_col).cast(T.DoubleType()))
    if 'num_class' in xgb_params:
        score = calculate_statistics(preds, label_col, multiclass=True, log=True)
    else:
        score = calculate_statistics(preds, label_col, log=True)
    return score


def return_suggest_categorical(trial, cfg, variable_name):
    return trial.suggest_categorical(name=cfg[variable_name], 
                                     choices=cfg[variable_name]['choices'])

def return_suggest_float(trial, cfg, variable_name):
    return trial.suggest_float(name=cfg[variable_name], 
                               low=cfg[variable_name]['low'],
                               high=cfg[variable_name]['high'],
                               log=cfg[variable_name]['log'],
                               )

def return_suggest_int(trial, cfg, variable_name):
    return trial.suggest_int(name=cfg[variable_name], 
                             low=cfg[variable_name]['low'],
                             high=cfg[variable_name]['high'],
                             log=cfg[variable_name]['log'],
                            )

def return_suggest_loguniform(trial, cfg, variable_name):
    return trial.suggest_int(name=cfg[variable_name], 
                             low=cfg[variable_name]['low'],
                             high=cfg[variable_name]['high']
                            )

def return_suggest_uniform(trial, cfg, variable_name):
    return trial.suggest_int(name=cfg[variable_name], 
                             low=cfg[variable_name]['low'],
                             high=cfg[variable_name]['high']
                            )


def get_default_params(cfg):
    def_xgb_params = {
            "eta": 0.1, "eval_metric": cfg['eval_metric'],
            "gamma": 1, "max_depth": 5, "min_child_weight": 1.0,
            "objective": cfg['objective'], "seed": 0,
            
            # xgboost4j only
            "num_round": 100, "num_early_stopping_rounds": 10,
            "maximize_evaluation_metrics": False,   # minimize logloss
            "num_workers": 1, "use_external_memory": False,
            "missing": np.nan,
        }    
    if cfg:
        for key in cfg['default_params'].keys():
            def_xgb_params[key] = cfg['default_params'][key] 
    return def_xgb_params
    
def suggest_by_type(cfg, trial):
    def_xgb_params = get_default_params()
    
    func_map = {
                    'categorical': return_suggest_categorical,
                    'float': return_suggest_float,
                    'int': return_suggest_int,
                    'loguniform': return_suggest_loguniform,
                    'uniform': return_suggest_uniform,
                }


    for category in cfg['search_space'].keys():
        if category in func_map.keys():
            for variable in cfg['search_space'][category].keys():
                def_xgb_params[variable] = func_map[category](trial, cfg['search_space'][category], variable)
        else:
            print(f'For {category} we were not able to provide transformation')

    return def_xgb_params

def optimize(train, valid, features_col, label_col, weight_col, cfg, spark):
    xgb_params = get_default_params()
    def objective(trial):
        suggested_params = suggest_by_type(cfg, trial)
        if int(suggested_params['num_class']) > 2:
            xgb_params['num_class'] = int(cfg['num_class'])
        score = cross_validate(train, valid, suggested_params, features_col, label_col, weight_col, spark, summary=False)
        logger_hyperparam.info('xgb_params in trial: %s', xgb_params)
        logger_hyperparam.info('score: %s', score)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=cfg['n_trials'])
    
    best_params = study.best_params
    for param in best_params.keys():
        xgb_params[param] = best_params[param]
    print(f'Best params {xgb_params}')
    return xgb_params


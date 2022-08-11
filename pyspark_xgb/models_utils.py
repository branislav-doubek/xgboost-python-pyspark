import logging
import pandas as pd
# import joblib
import pyspark.sql.functions as f
import numpy as np
from dataclasses import dataclass, asdict
from itertools import chain
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, QuantileDiscretizer
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from typing import Any, Dict, List, Union, cast
from pyspark.ml.wrapper import JavaWrapper


@dataclass
class DevXGBoostModelHyperparams:
    # General params
    nworkers: int = 3
    nthread: int = 1

    # Column Params
    featuresCol: str = 'features'
    labelCol: str = 'label'
    predictionCol: str = 'prediction'
    weightCol = 'weight'
    num_class: int = 2

    # Booster Params
    eval_metric: str = 'aucpr'
    objective: str = 'binary:logistic'
    seed: int = 42

    # Tree Booster Params
    #     alpha=0.0,
    colsample_bytree: float = 0.6  # uniform [0.3, 1]
    eta: float = 0.01  # logaritmic [0.01, 0.2]
    gamma: float = 15  # uniform [1,30]
    #     grow_policy='depthwise',
    #     max_bin=256,
    #     max_delta_step=0.0,
    max_depth: int = 30  # uniform [5, 30]
    min_child_weight: float = 1  # uniform [1, 50]
    #     reg_lambda=0.0,
    scale_pos_weight = 1.0
    #     sketch_eps=0.03,
    subsample: float = 0.6  # uniform [0.3., 0.7]
    #     tree_method="auto",


class DevXGBoostModel:
    def __init__(
            self,
            spark,
            hyperparams: DevXGBoostModelHyperparams,
            feature_cols: List[str],
            label_col: str,
            actions: List[str]
    ):
        self.parameters = asdict(hyperparams)
        self.feature_cols = feature_cols
        self.actions = actions
        self.label_col = label_col
        self.weights = None
        self.model = None
        self.vector_assembler = None
        self.spark = spark

    def initialize_model(self):
        scala_map = self.spark._jvm.PythonUtils.toScalaMap(self.parameters)
        java_obj = "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier"
        self.model = JavaWrapper._new_java_obj(java_obj, scala_map)\
                                .setFeaturesCol('features').setLabelCol(self.label_col)\
                                #.setWeightCol('weight')

    def calculate_weights(self, label_col):
        y_collect = label_col.groupBy(self.label_col).count().collect()
        unique_y = [x[self.label_col] for x in y_collect]
        total_y = sum([x['count'] for x in y_collect])
        unique_y_count = len(y_collect)
        bin_count = [x['count'] for x in y_collect]
        self.weights = {i: ii for i, ii in zip(unique_y, total_y / (
                    unique_y_count * np.array(bin_count)))}

    def weight_mapping(self, df: DataFrame):
        mapping_expr = f.create_map(
            [f.lit(x) for x in chain(*self.weights.items())])
        return df.withColumn('weight',
                             mapping_expr.getItem(f.col(self.label_col)))

    def create_feature_vector(self, df: DataFrame):
        return self.vector_assembler.transform(df)

    def train_vector_assembler(self, df: DataFrame):
        self.vector_assembler = VectorAssembler().setInputCols(
            self.feature_cols).setOutputCol('features')

    def predict(self, df: DataFrame):
        # cols = set(df.select(self.feature_cols).columns)
        # diff = cols.symmetric_difference(set(self.feature_cols))
        # if len(diff):
        #    raise ValueError(f"Columns are different: {diff}")
        self.train_vector_assembler(df)
        test = self.create_feature_vector(df)
        return self.model.transform(test._jdf)

    def save_model(self, path):
        jbooster = self.model.nativeBooster()
        jbooster.saveModel(model_path)

    def save_model(self, path):
        if os.path.exists(path):
            logger.info('load model from {}'.format(path))
            scala_xgb = self.spark.sparkContext._jvm.ml.dmlc.xgboost4j.scala.XGBoost
            jbooster = scala_xgb.loadModel(path)

            # uid, num_class, booster
            xgb_cls_model = JavaWrapper._new_java_obj(
                "ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel",
                "xgbc", 2, jbooster)
            self.model = xgb_cls_model


    def train_model(self, train_ds, valid_ds=False):
        self.initialize_model()
        if valid_ds:
            eval_set = {'eval': valid_ds._jdf}
            scala_eval_set = self.spark._jvm.PythonUtils.toScalaMap(eval_set)
            self.model = self.model.setEvalSets(scala_eval_set)
            self.model = self.model.fit(train_ds._jdf)
        else:
            self.model = self.model.fit(train_ds._jdf)

    def cross_validate(self, train_ds, valid_ds, multiclass=False) -> float:
        # Calculate weights
        self.calculate_weights(train_ds.select(self.label_col))
        train = self.weight_mapping(train_ds)
        valid = self.weight_mapping(valid_ds)
        # Create feature vectors
        self.train_vector_assembler(train_ds)
        train = self.create_feature_vector(train)
        valid = self.create_feature_vector(valid)
        # Fit model
        self.model = self.train_model(train, valid)
        # Predict (transform)
        predictions = self.model.transform(valid._jdf)
        predictions_labels = predictions.rdd.map(
            lambda x: (x['prediction'], x['label']))
        metrics = MulticlassMetrics(predictions_labels)

        # Statistics by class
        labels = predictions.rdd.map(lambda lp: lp.label).distinct().collect()
        score = 0
        for label in sorted(labels[1:]):
            logging.info(
                'Class %s precision = %s' % (label, metrics.precision(label)))
            logging.info(
                'Class %s recall = %s' % (label, metrics.recall(label)))
            logging.info('Class %s F1 Measure = %s' % (
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
        logging.info(score)

        cm = metrics.confusionMatrix()
        logging.info(cm)
        print('Confusion Matrix')
        print(cm)

        return score


def calculate_stats(spark, predictions):
    stats = pd.DataFrame()
    labels = predictions.rdd.map(lambda lp: lp.label).distinct().collect()
    for label in sorted(labels):
        probabilities = predictions.select('probabilities', 'prediction',
                                           'label').filter(
            f.col('prediction') == label).rdd.map(lambda row: (
            float(max(row['probabilities'])), float(row['prediction']),
            float(row['label'])))
        probabilities = spark.createDataFrame(
            probabilities, ['probabilities', 'prediction', 'label'])
        qds = QuantileDiscretizer(
            numBuckets=10,
            inputCol='probabilities',
            relativeError=0.01,
            handleInvalid='error'
        )
        bucketizer = qds.fit(probabilities)
        ntiles = bucketizer.setHandleInvalid('skip').transform(probabilities)
        # pdf = ntiles.groupBy('label', 'ntiles').agg({'label': 'count'}).toPandas()
        pdf = ntiles.groupBy('label', 'prediction', 'ntiles').agg(
            f.count('label').alias('cnt')).toPandas()
        stats = stats.append(pdf, ignore_index=True)
    return stats


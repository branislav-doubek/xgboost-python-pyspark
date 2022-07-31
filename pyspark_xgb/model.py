import logging
import pandas as pd
import joblib
import pyspark.sql.functions as F
import numpy as np
from dataclasses import dataclass, asdict
from itertools import chain
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, QuantileDiscretizer
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from typing import Any, Dict, List, Union, cast


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
            hyperparams: DevXGBoostModelHyperparams,
            model,
            feature_cols: List[str],
            label_col: str,
            actions: List[str]
    ):
        self.model = model(**asdict(hyperparams))
        self.feature_cols = feature_cols
        self.actions = actions
        self.label_col = label_col
        self.weights = None
        self.vector_assembler = None

    def calculate_weights(self, label_col):
        y_collect = label_col.groupBy(self.label_col).count().collect()
        unique_y = [x[self.label_col] for x in y_collect]
        total_y = sum([x['count'] for x in y_collect])
        unique_y_count = len(y_collect)
        bin_count = [x['count'] for x in y_collect]
        self.weights = {i: ii for i, ii in zip(unique_y, total_y / (
                    unique_y_count * np.array(bin_count)))}

    def weight_mapping(self, df: DataFrame):
        mapping_expr = F.create_map(
            [F.lit(x) for x in chain(*self.weights.items())])
        return df.withColumn('weight',
                             mapping_expr.getItem(F.col(self.label_col)))

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
        return self.model.transform(test)

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
        self.model = self.model.fit(train)
        # Predict (transform)
        predictions = self.model.transform(valid)
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

    # TODO Save model to JSON
    # def __getstate__(self):
    #     attributes = self.__dict__.copy()
    #     del attributes["model"]
    #     attributes["model_json"] = save_model_to_json(
    #         cast(XGBoostEstimator, self.model))
    #     return attributes
    #
    # def __setstate__(self, state):
    #     model_json = state.get("model_json", None)
    #     if model_json is None:
    #         raise ValueError(
    #             "Model doesn't include json data for xgboost model")
    #     del state["model_json"]
    #     self.__dict__ = state
    #     self.model = XGBoostEstimator()
    #     load_model_from_json(self.model, model_json)
    #
    # def save_model(self, fname: str):
    #     self.model.saveModelAsHadoopFile(fname)
    #
    # def get_model(self):
    #     return self.model

    @classmethod
    def load_model(cls, fname: str):
        return cast(DevXGBoostModel, joblib.load(fname))


def calculate_stats(spark, predictions):
    stats = pd.DataFrame()
    labels = predictions.rdd.map(lambda lp: lp.label).distinct().collect()
    for label in sorted(labels):
        probabilities = predictions.select('probabilities', 'prediction',
                                           'label').filter(
            F.col('prediction') == label).rdd.map(lambda row: (
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
            F.count('label').alias('cnt')).toPandas()
        stats = stats.append(pdf, ignore_index=True)
    return stats

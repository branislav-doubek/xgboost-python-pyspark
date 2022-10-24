import os
import traceback
import numpy as np
from pyspark.ml.feature import VectorAssembler
from spark import get_spark
from utils import create_feature_map, create_feature_imp, load_config
from utils import weight_mapping, train_model, load_model, predict, udf_logloss, save_model, optimize
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'

abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
PYSPARK_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
DATASET_PATH = '/home/dataset'
CONFIG_PATH = '/home/config.yml'
MODEL_PATH = '/output'
LOCAL_MODEL_PATH = MODEL_PATH

def main():

    try:
        config = load_config(CONFIG_PATH)
        spark = get_spark(app_name="pyspark-xgb")
        # load data
        train = spark.read.parquet(DATASET_PATH + '/train')
        valid = spark.read.parquet(DATASET_PATH + '/valid')
        safe_cols = config['safe_cols']
        LABEL = config['label_col']
        FEATURES = 'features'
        WEIGHT = 'weight'
        features = [c for c in train.columns if c not in safe_cols]
        print(features)
        assembler = VectorAssembler(inputCols=features, outputCol=FEATURES)
        train, weights = weight_mapping(train, LABEL)
        print(weights)
        print(train.show())
        valid = weight_mapping(valid, LABEL, weights)[0]
        print(valid.show())
        train = assembler.transform(train).select(FEATURES, LABEL, WEIGHT)
        valid = assembler.transform(valid).select(FEATURES, LABEL, WEIGHT)
        

        best_params = optimize(train, valid, FEATURES, LABEL, WEIGHT, config)
        jmodel = train_model(train, best_params, FEATURES, LABEL, WEIGHT)
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
        xgb_cls_model = load_model(local_model_path)
        pred = predict(xgb_cls_model, valid)
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




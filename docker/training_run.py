import subprocess
import yaml
from yaml import CLoader as Loader

def run_docker(config, training = False):

    if training:
        config_modeling = config['modeling']
        docker_img = 'o2sk_modeling_training'
    else:
        config_modeling = config['scoring']
        docker_img = 'o2sk_modeling_scoring'


    with open(config_modeling['docker_config_path'], 'w') as f:
        yaml.dump(config_modeling, f)
    if training:     
        subprocess.call(f"docker run -it --name o2sk_modeling_training -v {config_modeling['data_path']}:/home/dataset/ -v {config_modeling['docker_config_path']}:/home/config.yml -v {config_modeling['model_path']}:/output {docker_img}:latest", shell=True)
    else:
        subprocess.call(f"docker run -it --name o2sk_modeling_scoring -v {config_modeling['data_path']}:/home/dataset/ -v {config_modeling['docker_config_path']}:/home/config.yml -v {config_modeling['model_path']}:/output/model.bin -v {config_modeling['scoring_output_path']}:/data_output {docker_img}:latest", shell=True)

f = open('config.yml', 'r')
cfg = yaml.load(f, Loader)
run_docker(cfg, True)
# docker run -it -v <data_path>:/home/dataset/ -v <config_path>:/home/config.yml sparkxgb:latest /bin/bash
# docker run -it -v /home/frovis/o2_sk/xgboost-python-pyspark/docker/data/:/home/dataset -v /home/frovis/o2_sk/xgboost-python-pyspark/docker/config.yml:/home/config.yml xgb:latest /bin/bash  
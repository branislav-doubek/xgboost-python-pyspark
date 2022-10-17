import subprocess
import yaml


def run_docker(config, training: True, multiclass: False):

    if training:
        config_modeling = config['modeling']
        docker_img = 'o2sk_modeling_training'
    else:
        config_modeling = config['scoring']
        docker_img = 'o2sk_modeling_scoring'

    if multiclass:
        config_final = config_modeling['multiclass']
    else:
        config_final = config_modeling['binary']

    with open(config_final['config_path_docker'], 'w') as file:
        yaml.dump(config_final, indent=4)

    with open("/home/frovis/o2sk/xgboost-python-pyspark/docker/output.log", "a") as output:
        subprocess.call(f"docker run -it --name sk_training_v1 -v {config_final['data_path']}:/home/dataset/ -v {config_final['config_path_docker']}:/home/config.yml -v sk_modeling:/output {docker_img}:latest", shell=True, stdout=output, stderr=output)
        #subprocess.call("CID=$(docker run -d -v sk_modeling:/output busybox true)", shell=True, stdout=output, stderr=output)
        subprocess.call("docker cp sk_training_v1:/output ./", shell=True, stdout=output, stderr=output)


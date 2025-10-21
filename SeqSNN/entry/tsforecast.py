import warnings
import os
from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from SeqSNN.dataset import DATASETS
from SeqSNN.runner import RUNNERS
from SeqSNN.network import NETWORKS

import argparse
import yaml  # 用于加载YAML配置文件


warnings.filterwarnings("ignore")


@configclass
class SeqSNNConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    runner: RegistryConfig[RUNNERS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(dataset_name="train")
    validset = config.data.build(dataset_name="valid")
    testset = config.data.build(dataset_name="test")
    network = config.network.build(
        input_size=trainset.num_variables, max_length=trainset.max_seq_len
    )
    runner = config.runner.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=config.runner.out_size or trainset.num_classes,
    )
    runner.fit(trainset, validset, testset)
    runner.predict(trainset, "train")
    runner.predict(validset, "valid")
    runner.predict(testset, "test")


if __name__ == "__main__":
    # # 硬编码配置文件路径
    # # config_file_path = "exp/forecast/ispikformer/ispikformer_electricity.yml"
    #
    # config_file_path = "/home/dan/zmz/timeserise/SeqSNN/SeqSNN-main/exp/forecast/tcn/spiketcn2d_pems-bay.yml"
    # # 加载 YAML 配置文件
    # with open(config_file_path, "r") as config_file:
    #     config_data = yaml.safe_load(config_file)
    #
    # # _config = SeqSNNConfig.from_dict(config_data)
    # # _config = SeqSNNConfig.fromcli(config_data)
    # # 手动映射配置字典到 SeqSNNConfig
    # _config = SeqSNNConfig(
    #     data=config_data.get('data'),
    #     network=config_data.get('network'),
    #     runner=config_data.get('runner'),
    #     runtime=config_data.get('runtime', RuntimeConfig())
    # )
    # # 使用加载的配置运行训练过程
    # run_train(_config)
    #
    #
    _config = SeqSNNConfig.fromcli()
    run_train(_config)
    #

    # # 硬编码配置文件路径
    # config_file_path = "/home/dan/zmz/timeserise/SeqSNN/SeqSNN-main/exp/forecast/spikegru/spikegru_pems-bay.yml"
    #
    # # 检查配置文件是否存在
    # if not os.path.exists(config_file_path):
    #     print(f"错误: 找不到配置文件 '{config_file_path}'")
    #     exit(1)
    #
    # # 从指定的配置文件加载配置
    # _config = SeqSNNConfig.fromfile(config_file_path)
    # run_train(_config)
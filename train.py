from __future__ import absolute_import

import argparse
import boto3
import copy
import mmcv
import os
import os.path as osp
import random as rd
import subprocess
import sys
import time
import time
import torch
import torch.distributed as dist
import warnings

from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_device, get_root_logger, setup_multi_processes

from utils import print_files_in_path, save_model_artifacts

ROLE_ARN = os.environ.get('ROLE_ARN')
ROLE_NAME = os.environ.get('ROLE_NAME')


def assumed_role_session():
    # Assume the "notebookAccessRole" role we created using AWS CDK.
    client = boto3.client('sts')
    # creds = client.assume_role(RoleArn=ROLE_ARN, RoleSessionName=ROLE_NAME)['Credentials']
    return boto3.session.Session(
        # aws_access_key_id=creds['AccessKeyId'],
        # aws_secret_access_key=creds['SecretAccessKey'],
        # aws_session_token=creds['SessionToken'],
        # region_name='us-east-1',
    )


def download_data(data, split):
    split_folder = f"/opt/ml/data/{split}"
    if not (os.path.exists(split_folder)):
        os.makedirs(split_folder)
    session = assumed_role_session()
    s3_connection = session.resource('s3')
    splits = data.split('/')
    bucket_name = splits[2]
    bucket = s3_connection.Bucket(bucket_name)
    objects = list(bucket.objects.filter(Prefix="/".join(splits[3:] + [split])))
    print("Downloading files.")
    for iter_object in objects:
        splits = iter_object.key.split('/')
        if splits[-1]:
            filename = f"{split_folder}/{splits[-1]}"
            bucket.download_file(iter_object.key, filename)
    print("Finished downloading files.")


def train():
    # print("\nList of files in train channel: ")
    # print_files_in_path(os.environ.get("SM_CHANNEL_TRAIN"))

    # print("\nList of files in validation channel: ")
    # print_files_in_path(os.environ.get("SM_CHANNEL_VALIDATION"))

    # print("\nList of files in Test channel: ")
    # print_files_in_path(os.environ.get("SM_CHANNEL_TEST"))

    config_file = os.environ.get('CONFIG_FILE')
    print(f'\n config file: {config_file}')

    print(f"Environment variables: {os.environ}")

    # Dummy net.
    net = None

    # download and prepare data for training:
    for split in ['training', 'validation', 'test', 'configs', 'models']:
        download_data(os.environ.get('S3_URL'), split)

    # for root, dirs, files in os.walk("/opt/ml/data", topdown=False):
    #     for name in files:
    #         print("files:", os.path.join(root, name))
    #     for name in dirs:
    #         print("files:", os.pvimath.join(root, name))

    # subprocess.run(f"mim train mmsegmentation {config_file} --launcher pytorch --gpus 1".split(' '))
    # set random seeds
    cfg = Config.fromfile(config_file)
    cfg.device = get_device()
    seed = init_random_seed(10, device=cfg.device)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model_path = f"/opt/ml/code/{os.environ['VERSION']}/{os.environ['EVENT_TYPE']}/"
    os.makedirs(model_path)

    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed, deterministic=True)
    cfg.seed = seed

    os.environ['RANK'] = str(torch.cuda.device_count())
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_PORT'] = str(rd.randint(20000, 30000))
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_file))[0])
    cfg.gpu_ids = range(1)

    distributed = False
    print('#######', cfg.dist_params)

    # init_dist('pytorch', **cfg.dist_params)

    # gpu_ids is used to calculate iter when resuming checkpoint
    # _, world_size = get_dist_info()
    # cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config_file)))

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config_file)

    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.'
        )
        model = revert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model, datasets, cfg, distributed=distributed, validate=True, timestamp=timestamp, meta=meta
    )

    # At the end of the training loop, we have to save model artifacts.
    model_dir = os.environ["MODEL_DIR"]
    session = assumed_role_session()
    s3_connection = session.resource('s3')
    save_model_artifacts(s3_connection, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # sagemaker-containers passes hyperparameters as arguments
    parser.add_argument("--hp1", type=str)
    parser.add_argument("--hp2", type=int, default=50)
    parser.add_argument("--hp3", type=float, default=0.1)

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args()

    train()

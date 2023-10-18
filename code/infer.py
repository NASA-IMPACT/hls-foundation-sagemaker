print("!!!! importing packages")
import os
import matplotlib.pyplot as plt
import mmcv
import gc
import logging

import rasterio
import torch

from mmseg.datasets.pipelines.compose import Compose

from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from post_process import PostProcess

print("!!!! Done importing packages")

CONFIG_DIR = "/opt/mmsegmentation/configs/{experiment}_config/geospatial_fm_config.py"
DOWNLOAD_FOLDER = "/opt/downloads"

MODEL_CONFIGS = {
    'flood': {
        'config': 'sen1floods11_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-sen1floods11',
        'weight': 'sen1floods11_Prithvi_100M.pth',
        'collections': ['HLSS30'],
    },
    'burn_scars': {
        'config': 'burn_scars_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-burn-scar',
        'weight': 'burn_scars_Prithvi_100M.pth',
        'collections': ['HLSS30', 'HLSL30'],
    },
    # 'crop_classification': {
    #     'config': 'multi_temporal_crop_classification_Prithvi_100M.py',
    #     'repo': 'ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification',
    #     'weight': 'multi_temporal_crop_classification_Prithvi_100M.pth',
    #     'collections': ['HLSS30', 'HLSL30'],
    # },
}

def update_config(config, model_path):
    with open(config, 'r') as config_file:
        config_details = config_file.read()
        updated_config = config_details.replace('<path to pretrained weights>', model_path)

    with open(config, 'w') as config_file:
        config_file.write(updated_config)


def load_model(model_name):
    repo = MODEL_CONFIGS[model_name]['repo']
    config = hf_hub_download(repo, filename=MODEL_CONFIGS[model_name]['config'])
    model_path = hf_hub_download(repo, filename=MODEL_CONFIGS[model_name]['weight'])
    update_config(config, model_path)
    infer = Infer(config, model_path)
    _ = infer.load_model()
    return infer


class Loader:
    def __init__(self):
        self.initialized = False
        self.models = {}
        
    # copied over from https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/multi_model_bring_your_own/container/model_handler.py
    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_path = properties.get("model_dir")
        print('!!!!!!!', model_path, MODEL_CONFIGS)
        self.models = { model_name: load_model(model_name) for model_name in MODEL_CONFIGS }
        self.initialized = True
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        print(f"****************{context}, {data}****************")
        model_out = self.models[context.system_properties['model_dir']].infer([data])
        return infer.postprocess([model_out], [data])


class Infer:
    def __init__(self, config, checkpoint):
        self.initialized = False
        self.config_filename = config
        self.checkpoint_filename = checkpoint

    # copied over from https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/multi_model_bring_your_own/container/model_handler.py
    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_path = properties.get("model_dir")
        checkpoint_filename = context.model_name
        logging.info("Model directory: {}, {}".format(checkpoint_filename, properties))
        logging.info(f"Configuration::: {list(os.walk(model_path))}" )
        # gpu_id = properties.get("gpu_id")
        self.load_model_config_file(model_path, checkpoint_filename)
        # load models here 
        # load model
        self.load_model()

    def load_model(self):
        self.config = mmcv.Config.fromfile(self.config_filename)
        self.config.model.pretrained = None
        self.config.model.train_cfg = None

        if self.checkpoint_filename is not None:
            self.model = init_segmentor(self.config, self.checkpoint_filename, device="cuda:0")
            self.checkpoint = load_checkpoint(
                self.model, self.checkpoint_filename, map_location="cpu"
            )
            self.model.CLASSES = self.checkpoint["meta"]["CLASSES"]
            self.model.PALETTE = self.checkpoint["meta"]["PALETTE"]
        self.model.cfg = self.config  # save the config in the model for convenience
        self.model.to("cuda:0")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        return self.model

    def infer(self, images):
        """
        Infer on provided images
        Args:
            images (list): List of images
        """
        test_pipeline = self.config.data.test.pipeline
        test_pipeline = Compose(test_pipeline)
        data = []
        for image in images:
            image_data = dict(img_info=dict(filename=image))
            image_data['seg_fields'] = []
            image_data['img_prefix'] = DOWNLOAD_FOLDER
            image_data['seg_prefix'] = DOWNLOAD_FOLDER
            image_data = test_pipeline(image_data)
            data.append(image_data)
        data = collate(data, samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            data["img_metas"] = [i.data[0] for i in list(data["img_metas"])]

        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        return result

    def load_model_config_file(self, model_path, model_name):
        """
        Get the model config based on the selected model filename.
        This assume config exist in the CONFIG_DIR

        :param model_dir: Path to the directory with model artifacts
        :return: model config file
        """
        from glob import glob
        model_files = glob(f"{model_path}/*")
        logging.info(f"Configuration:: {model_files}")
        model_name = model_files[-1]
        splits = os.path.basename(model_name).replace('.pth', '').split('_')
        username = splits[0] 
        experiment = "_".join(splits[1:])

        self.config_filename = CONFIG_DIR.format(experiment=experiment)
        self.checkpoint_filename = f"{model_path}/{model_name}"
        logging.info("Model config for user {}: {}".format(username, self.config_filename))

    def postprocess(self, results, files):
        """
        Postprocess results to prepare geojson based on the images

        :param results: list of results from infer method
        :param files: list of files on which the inference was performed

        :return: GeoJSON of detected features
        """
        transforms = list()
        geojson_list = list()
        for tile in files:
            with rasterio.open(tile) as raster:
                transforms.append(raster.transform)
        for index, result in enumerate(results):
            detections = PostProcess.extract_shapes(result, transforms[index])
            detections = PostProcess.remove_intersections(detections)
            geojson = PostProcess.convert_to_geojson(detections)
            for geometry in geojson:
                updated_geometry = PostProcess.convert_geojson(geometry)
                geojson_list.append(updated_geometry)
        return {"type": "FeatureCollection", "features": geojson_list}


print("!!!!!", "Loading models")
_service = Loader()

def handle(data, context):
    logging.info(f"Data to be processed: {data}")
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None
    return _service.handle(data, context)

import gc
import os
import rasterio
import time
import torch

from app.lib.downloader import Downloader
from app.lib.infer import Infer
from app.lib.post_process import PostProcess

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from huggingface_hub import hf_hub_download

from multiprocessing import Pool, cpu_count

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

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


MODELS = {model_name: load_model(model_name) for model_name in MODEL_CONFIGS}


def download_files(infer_date, layer, bounding_box):
    downloader = Downloader(infer_date, layer)
    return downloader.download_tiles(bounding_box)


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {'Hello': 'World'}


@app.get('/models')
def list_models():
    response = jsonable_encoder(list(MODEL_CONFIGS.keys()))
    return JSONResponse({'models': response})


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def infer_from_model(request: Request):
    body = await request.json()

    instances = body['instances'][0]
    model_id = instances['model_id']
    infer_date = instances['date']
    bounding_box = instances['bounding_box']

    if model_id not in MODELS:
        response = {'statusCode': 422}
        return JSONResponse(content=jsonable_encoder(response))
    infer = MODELS[model_id]
    all_tiles = list()
    geojson_list = list()
    download_infos = list()

    for layer in MODEL_CONFIGS[model_id]['collections']:
        download_infos.append((infer_date, layer, bounding_box))

    pool = Pool(cpu_count() - 1)
    start_time = time.time()
    all_tiles = pool.starmap(download_files, download_infos)
    all_tiles = [tile for tiles in all_tiles for tile in tiles]
    pool.close()
    pool.join()
    print("!!! Download Time:", time.time() - start_time)

    start_time = time.time()
    if all_tiles:
        results = infer.infer(all_tiles)
        transforms = list()
        for tile in all_tiles:
            with rasterio.open(tile) as raster:
                transforms.append(raster.transform)
        for index, result in enumerate(results):
            detections = PostProcess.extract_shapes(result, transforms[index])
            detections = PostProcess.remove_intersections(detections)
            geojson = PostProcess.convert_to_geojson(detections)
            for geometry in geojson:
                updated_geometry = PostProcess.convert_geojson(geometry)
                geojson_list.append(updated_geometry)
    print("!!! Infer Time:", time.time() - start_time)
    del infer
    gc.collect()
    torch.cuda.empty_cache()
    final_geojson = {'predictions': [{'type': 'FeatureCollection', 'features': geojson_list}]}
    return JSONResponse(content=jsonable_encoder(final_geojson))

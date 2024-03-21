import boto3
import os
from os import path
from glob import glob

BUCKET_NAME = os.environ['S3_URL'].split('/')[2]
MODEL_PATH = "models/{model_name}"


def save_model_artifacts(s3_connection, model_artifacts_path):
    if path.exists(model_artifacts_path):
        for model_file in glob(f"{model_artifacts_path}/*.pth"):
            model_name = model_file.split('/')[-1]
            model_name = os.environ.get('MODEL_NAME', model_name)
            model_name = MODEL_PATH.format(model_name=model_name)
            print(f"Uploading model to s3: s3://{BUCKET_NAME}/{model_name}")
            s3_connection.meta.client.upload_file(model_file, BUCKET_NAME, model_name)


def print_files_in_path(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        print(f)

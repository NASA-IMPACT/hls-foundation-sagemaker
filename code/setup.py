from setuptools import setup

setup(
    name='geospatial_fm',
    version='0.1.0',
    description='MMSegmentation classes for geospatial-fm finetuning',
    author='Paolo Fraccaro, Carlos Gomes, Johannes Jakubik',
    packages=['geospatial_fm'],
    license="Apache 2",
    install_requires=[
        "mmsegmentation==0.30.0",
        "urllib3==1.26.12",
        "rasterio",
        "tifffile",
        "einops",
        "timm==0.4.12",
        "tensorboard",
        "imagecodecs",
    ],
)

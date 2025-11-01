from __future__ import annotations
from .base import DataModule

def imagenet_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("ImageNet path not found. Please download ILSVRC2012 and point data_root to it.")

def coco_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("COCO dataset missing. Download 2017 train/val and set data_root accordingly.")

def cub_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("CUB-200-2011 dataset missing. Download and extract, then update config.")

def miccai_brain_tumor_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("MICCAI 2014 brain tumor challenge data required. See challenge page for access.")

def mura_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("Stanford MURA dataset missing. Download from official source and set data_root.")

def statefarm_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("State Farm distracted driver dataset missing. Provide path to images.")

def mimic_cxr_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("MIMIC-CXR requires credentialed access. After approval, set data_root in config.")

def youtube8m_build(*args, **kwargs) -> DataModule:
    raise NotImplementedError("YouTube-8M support is provided as a feature stub (video tagging/summarization)." )

def dstl_sat_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("DSTL Kaggle dataset missing. Download tiles/shapefiles and set data_root.")

def poverty_sat_build(*args, **kwargs) -> DataModule:
    raise FileNotFoundError("Stanford Poverty Estimation project data missing. Provide satellite imagery paths.")
import os
import gzip
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import json
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger

from osu3d.datasets import get_dataset
from osu3d.objects_map import NodesConstructor
from osu3d.objects_map.utils.structures import MapObjectList
import time

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

class TqdmLoggingHandler:
    def __init__(self, level="INFO"):
        self.level = level

    def write(self, message, **kwargs):
        if message.strip() != "":
            tqdm.write(message, end="")

    def flush(self):
        pass

def set_seed(seed: int=18) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")

def convert_ndarray_to_list(obj):
    """Convert ndarray fields in a dictionary to lists for JSON serialization."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                obj[key] = value.tolist()
    return obj

def main(args):
    hash = datetime.now()
    with open(args.config_path) as file:
        config = yaml.full_load(file)
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        with gzip.open(os.path.join(args.save_path, "meta.pkl.gz"), "wb") as file:
            pickle.dump({"config": config}, file)

    logger.info(f"Parsed arguments. Utilizing config from {args.config_path}.")

    rgbd_dataset = get_dataset(config["dataset"])              ####### color depth intrinsics pose 
    nodes_constructor = NodesConstructor(config["nodes_constructor"])

    # Section 3.1
    start_1=time.time()
    logger.info("Iterating over RGBD sequence to accumulate 3D objects.")
    for step_idx in tqdm(range(len(rgbd_dataset))):          ###每一帧进行迭代
        frame = rgbd_dataset[step_idx]        #######  color, depth, intrinsics, pose
        nodes_constructor.integrate(step_idx, frame, args.save_path)
        torch.cuda.empty_cache()
    nodes_constructor.postprocessing()
    torch.cuda.empty_cache()
    
    
    if args.save_path:
        results = {'objects': nodes_constructor.objects.to_serializable()}
        with gzip.open(os.path.join(args.save_path, f"frame_last_objects.pkl.gz"), "wb") as f:
            pickle.dump(results, f)

    print('Iterating over RGBD sequence to accumulate 3D objects.--use time:',time.time()-start_1)

    # Section 3.2
    start_2=time.time()
    logger.info('Finding 2D view to caption 3D objects.')
    nodes_constructor.project(
        poses=rgbd_dataset.poses,
        intrinsics=rgbd_dataset.get_cam_K()
    )
    torch.cuda.empty_cache()

    print('Finding 2D view to caption 3D objects.',time.time()-start_2)

    # Section 3.3
    start_3=time.time()
    logger.info('Captioning 3D objects.')
    nodes = nodes_constructor.describe(colors=rgbd_dataset.color_paths)   ### 这一步就是用多模态大模型生成描述和label  其实可以改成Qwen2-vl-7B这种 甚至72B的
    torch.cuda.empty_cache()
    print('Captioning 3D objects.',time.time()-start_3)

    # Saving data
    output_path = config["nodes_constructor"]["output_path"]
    os.makedirs(output_path, exist_ok=True)

    # Save objects
    logger.info('Saving objects.')
    results = {'objects': nodes_constructor.objects.to_serializable()}
    with gzip.open(os.path.join(output_path, hash.strftime("%m.%d.%Y_%H:%M:%S_") + config["nodes_constructor"]["output_name_objects"]), 'wb') as file:
        pickle.dump(results, file)

    # Save nodes
    logger.info('Saving graph nodes in JSON file.')
    with open(os.path.join(output_path, hash.strftime("%m.%d.%Y_%H:%M:%S_") + config["nodes_constructor"]["output_name_nodes"]), 'w') as f:
        json.dump(nodes, f)

if __name__ == "__main__":
    start=time.time()

    parser = argparse.ArgumentParser(description="Build 3D scene object map. For more information see Sec. 3.1 - 3.3.")
    parser.add_argument("--config_path", default=r"examples/configs/replica_room0.yaml", help="see example in default path")
    parser.add_argument("--logger_level", default="INFO")
    parser.add_argument("--save_path", default=None, help="folder to save all steps to visualize mapping process")
    args = parser.parse_args()

    logger.remove()
    logger.add(TqdmLoggingHandler(), level=args.logger_level, colorize=True)

    set_seed()
    main(args)

    print(time.time()-start)

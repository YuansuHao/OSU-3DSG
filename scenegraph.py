import os
import gzip
import pickle
import json
import torch
torch.set_grad_enabled(False)
from loguru import logger
from datetime import datetime
import yaml
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm
from PIL import Image
import numpy as np
from itertools import combinations
from qwen_vl_utils import process_vision_info
import time 
'''
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass
'''
def load_dataset(json_path):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    return dataset

class TripletConstructor:
    def __init__(self, config):
        # Initialize and load Qwen model
        self.model_path = config["model_path"]
        self.device = config.get("device", "cuda")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            local_files_only=True
        )
        logger.info(f"Qwen Model loaded successfully!")

    def resize_image_and_bbox(self, image_path, bbox_center, bbox_extent, max_size=600):
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Get the original image size
        original_width, original_height = image.size
        
        # Calculate the scaling factor for resizing
        scale_factor = max_size / max(original_width, original_height)
        
        # Calculate the new size while maintaining aspect ratio
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Scale bbox_center and bbox_extent
        scaled_bbox_center = [coord * scale_factor for coord in bbox_center]
        scaled_bbox_extent = [extent * scale_factor for extent in bbox_extent]
        
        return resized_image, scaled_bbox_center, scaled_bbox_extent

    def extract_relations(self, objects, color_images_path):
        triplets = []
        object_pairs = self.find_object_pairs(objects)
        semantic_pair_count  = 0
        # 计算每个对象对的距离并存储
        distances = []
        for subject, object_ in object_pairs:
            distance = self.calculate_distance(subject["bbox_center"], object_["bbox_center"])
            distances.append((subject, object_, distance))
        # 按距离排序
        distances.sort(key=lambda x: x[2])

        # 前50%距离为 medium，后50%距离为 far
        num_pairs = len(distances)
        threshold_index = num_pairs // 2

        for idx, (subject, object_, distance) in enumerate(distances):
            torch.cuda.empty_cache()

            #subject_label = self.extract_label(subject["description"])
            #object_label = self.extract_label(object_["description"])
            #distance = self.calculate_distance(subject["bbox_center"], object_["bbox_center"])

            # 路径设置
            # Replica处理方式
            # subject_image_path = os.path.join(color_images_path, f"frame{subject['color_image_idx']:06d}.jpg")
            # 3RScan处理方式
            # subject_image_path = os.path.join(color_images_path, f"frame-{subject['color_image_idx']:06d}.color.jpg")
            # ScanNet处理方式
            subject_image_path = os.path.join(color_images_path, f"{subject['color_image_idx']}.jpg")

            subject_image = subject_image_path  # 使用不缩放的图像
            scaled_subject_center, scaled_subject_extent = subject["bbox_center"], subject["bbox_extent"]
            scaled_object_center, scaled_object_extent = object_["bbox_center"], object_["bbox_extent"]
            # 缩放
            # subject_image, scaled_subject_center, scaled_subject_extent = self.resize_image_and_bbox(
            #     subject_image_path, subject["bbox_center"], subject["bbox_extent"]
            # )
            
            # object_image_path = os.path.join(color_images_path, f"frame{object_['color_image_idx']:06d}.jpg")
            # _, scaled_object_center, scaled_object_extent = self.resize_image_and_bbox(
            #     object_image_path, object_["bbox_center"], object_["bbox_extent"]
            # )

            # 构建 subject 和 object 信息
            subject_info = {
                'image': subject_image,
                'bbox_extent': scaled_subject_extent,
                'bbox_center': scaled_subject_center,
                'description': subject["description"]
            }
            object_info = {
                'bbox_extent': scaled_object_extent,
                'bbox_center': scaled_object_center,
                'description': object_["description"]
            }
            relation_info = {'distance': distance}

            # 计算 IoU
            iou = self.compute_3d_iou(
                subject["bbox_center"], subject["bbox_extent"],
                object_["bbox_center"], object_["bbox_extent"]
            )
            
            # 设置 spatial 关系
            if iou > 0:
                spatial = "close"
            elif idx < threshold_index:
                spatial = "medium"
            else:
                spatial = "far"

            # 获取 semantic 关系
            semantic = None
            answer = None
            if iou > 0 and distance < 0.5 and idx<50:
                semantic_pair_count += 1
                print(f"iou:{iou}, distance:{distance}")
                start=time.time()
                answer = self.describe_relation(subject_info, object_info, relation_info, subject_image)
                print(time.time()-start)
                if answer:
                    semantic = answer[1]  # Qwen 生成的 semantic
            
            # 构建三元组
            triplets.append({
                "idx": idx,
                "subject": {
                    "id": subject["id"],
                    "label": answer[0] if answer else None,
                    "bbox_extent": subject["bbox_extent"],
                    "bbox_center": subject["bbox_center"],
                    "description": subject["description"],
                    "color_image_idx": subject["color_image_idx"]
                },
                "object": {
                    "id": object_["id"],
                    "label": answer[2] if answer else None,
                    "bbox_extent": object_["bbox_extent"],
                    "bbox_center": object_["bbox_center"],
                    "description": object_["description"],
                    "color_image_idx": object_["color_image_idx"]
                },
                "semantic": semantic,
                "spatial": {
                    "distance": distance,
                    "level": spatial
                }
            })
            
        print(f"Total number of semantic pairs: {semantic_pair_count}")

        return triplets

    def compute_3d_iou(self, bbox1_center, bbox1_extent, bbox2_center, bbox2_extent, padding=0.0, use_iou=True):
        padding = float(padding)

        # 计算边界框的最小和最大坐标
        bbox1_min = np.array(bbox1_center) - np.array(bbox1_extent) / 2 - padding
        bbox1_max = np.array(bbox1_center) + np.array(bbox1_extent) / 2 + padding

        bbox2_min = np.array(bbox2_center) - np.array(bbox2_extent) / 2 - padding
        bbox2_max = np.array(bbox2_center) + np.array(bbox2_extent) / 2 + padding

        # 计算两个边界框之间的重叠区域
        overlap_min = np.maximum(bbox1_min, bbox2_min)
        overlap_max = np.minimum(bbox1_max, bbox2_max)
        overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

        # 计算重叠体积和各边界框的体积
        overlap_volume = np.prod(overlap_size)
        bbox1_volume = np.prod(bbox1_max - bbox1_min)
        bbox2_volume = np.prod(bbox2_max - bbox2_min)
        total_volume = bbox1_volume + bbox2_volume - overlap_volume

        # 计算 IoU 或最大重叠率
        iou = overlap_volume / total_volume if total_volume > 0 else 0
        return iou if use_iou else max(overlap_volume / bbox1_volume, overlap_volume / bbox2_volume)

    def find_object_pairs(self, objects):
        object_pairs = list(combinations(objects, 2))
        logger.info(f"Number of potential object pairs: {len(object_pairs)}")
        return object_pairs

    def get_prompt(self, subject_info, object_info, relation_info):
        distance = relation_info.get('distance', 'unknown')
        iou = relation_info.get('iou', 'unknown')
        prompt = f"""
            Here are two objects.
            The descriptions of the two objects are as follows: {subject_info['description']} for the first object, and {object_info['description']} for the second.
            The bounding box of the first object has a center at {subject_info['bbox_center']} and an extent of {subject_info['bbox_extent']}.
            The bounding box of the second object has a center at {object_info['bbox_center']} and an extent of {object_info['bbox_extent']}.
            The two objects' distance between them is {distance}.
            
            Please analyze the description of the object, reasoning what this object actually is, and extract the noun as the label of the object. The noun can only be one word.
            For example, Extract 'seat' from 'A brown leather seat' as the label of the object. 
            If the description is so abstract such as 'a black rectangular object' that you can not extract a suitable noun as its label, please just the description be the label.

            Then, Choose the one you think best describes the semantic relationship between the two objects from the following relationships as the relationship:
            'supported by;left;right;front;behind;close by;inside;bigger than;smaller than;higher than;lower than;same symmetry as;same as;attached to;standing on;lying on;hanging on;connected to;leaning against;part of;belonging to;build in;standing in;cover;lying in;hanging in'

            Ensure that 'subject label', 'relationship', and 'object label' can form a coherent sentence when combined.
            For example, 'subject label': TV, 'relationship': is close to, 'object label': desk, and the sentence is 'TV is close to desk'.
            
            Please output the result as a list of three elements in the following format:
            ["subject label", "relationship", "object label"]
            """
        return prompt
    
    def describe_relation(self, subject_info, object_info, relation_info, image):
        torch.cuda.empty_cache()  

        prompt = self.get_prompt(subject_info, object_info, relation_info)
        
        # Prepare the message content in the required format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template for the text and process the vision info
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = process_vision_info(messages)
        
        image_inputs = image_inputs[0]
        
        # Process inputs using the processor
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device, dtype=torch.float16)  

        # Generate model output using the inputs
        output_ids = self.model.generate(**inputs, max_new_tokens=32)  
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        if isinstance(output_text, list) and len(output_text) > 0:
            output_text = output_text[0]

        print(output_text)

        try:
            answer = eval(output_text) if isinstance(output_text, str) else output_text
            if isinstance(answer, list) and len(answer) == 3:
                return answer
            else:
                logger.warning(f"Unexpected format in output: {output_text}")
                return None
        except Exception as e:
            logger.warning(f"Failed to parse output as list: {output_text}, error: {e}")
            return None

    def calculate_distance(self, center1, center2):
        # Calculate Euclidean distance between two bounding box centers
        center1 = np.array(center1)
        center2 = np.array(center2)
        return np.linalg.norm(center1 - center2)

def main(args):
    hash = datetime.now()
    with open(args.config_path) as file:
        config = yaml.full_load(file)
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        with gzip.open(os.path.join(args.save_path, "meta.pkl.gz"), "wb") as file:
            pickle.dump({"config": config}, file)
        
    logger.info(f"Parsed arguments. Utilizing config from {args.config_path}.")

    # Load dataset and output
    json_path = config["output"]["json_path"]
    rgbd_dataset = load_dataset(json_path)

    # Extract triplet relations
    logger.info('Constructing triplets.')
    triplet_constructor = TripletConstructor(config["triplet_constructor"])
    triplets = triplet_constructor.extract_relations(
        rgbd_dataset, config["dataset"]["color_images_path"]
    )
    
    output_dir = config["output"]["output_path"]
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{timestamp}_{config['output']['output_name']}")

    results = {
        'relations': triplets
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    s=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--save_path', type=str, required=False, help='Path to save output files.')
    args = parser.parse_args()
    main(args)
    print(time.time()-s)

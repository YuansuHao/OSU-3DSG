import numpy as np
from loguru import logger

from osu3d.objects_map.utils.structures import MapObjectList
from osu3d.objects_map.utils.objects import merge_obj2_into_obj1, \
    process_pcd, get_bounding_box, merge_objects, compute_overlap_matrix


def merge_objects_postprocessing(objects: MapObjectList, bool_matrix, downsample_voxel_size):
    x, y = bool_matrix.nonzero()
    bool_matrix = bool_matrix[bool_matrix]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, bool_matrix):
        if ratio :
            if kept_objects[j]:
                objects[j] = merge_obj2_into_obj1(
                    objects[j], objects[i], downsample_voxel_size, are_objects=True)
                kept_objects[i] = False
        else:
            break
    
    # Remove the objects that have been merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    return objects

def denoise_objects(objects: MapObjectList, downsample_voxel_size, dbscan_remove_noise,
                    dbscan_eps, dbscan_min_points):
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], downsample_voxel_size,
                                dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True)[0]
        if len(objects[i]['pcd'].points) < 20:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(objects[i]['pcd'])
        objects[i]['bbox'].color = [0, 1, 0]
    return objects

def postprocessing(objects, config):
    '''
    对累积的 3D 物体列表进行后处理
    · 过滤：去除点云数据不足、检测次数太少的物体
    · 去噪：使用 DBSCAN 算法去除噪声点云
    · object合并：基于物体的空间位置和视觉相似度，合并重复检测的物体
    · 空间合并：进一步基于物体之间的空间重叠进行优化，精简物体列表
    · 高度过滤：最后移除位于较高位置的天花板物体
    '''
    logger.info(f"Before postprocessing: {len(objects)} objects")

    # 基本过滤
    logger.debug("Start filtering") 
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= config["postprocessing"]["obj_min_points"] and \
           obj['num_detections'] >= config["postprocessing"]["obj_min_detections"]:
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    logger.debug(f"After basic filtering: {len(objects)}")

    # 去噪处理
    logger.debug("Start denoising")
    objects = denoise_objects(objects,
        config["detections_assembler"]["downsample_voxel_size"],
        True,
        config["detections_assembler"]["dbscan_eps"],
        config["detections_assembler"]["dbscan_min_points"])
    logger.debug(f"After denoising: {len(objects)}")

    # 物体合并
    logger.debug("Start merging")
    _objects_count = 0
    while (_objects_count != len(objects)):
        if _objects_count != 0:
            logger.debug("Repeating merging step")
        _objects_count = len(objects)
        objects = merge_objects(objects,
            config["objects_associator"]["merge_objects_overlap_thresh"],
            config["objects_associator"]["merge_objects_visual_sim_thresh"],
            config["detections_assembler"]["downsample_voxel_size"])
    logger.debug(f"After merging: {len(objects)}")

    # 空间合并
    logger.debug("Start spatial merging postprocess")
    overlap_matrix = compute_overlap_matrix(objects,
        config["detections_assembler"]["downsample_voxel_size"])
    matrix_bool = overlap_matrix > 0.3
    matrix_bool = matrix_bool * matrix_bool.T
    objects = merge_objects_postprocessing(objects, matrix_bool,
        config["detections_assembler"]["downsample_voxel_size"])
    logger.debug(f"After spatial merging: {len(objects)}")

    # # 最终高度过滤（移除天花板）
    # logger.debug("Final height filtering for ceiling objects")
    # objects_to_keep = []
    # for obj in objects:
    #     if obj['bbox'].center[2] <= 1.2:  # 筛选高度不超过1.2的对象
    #         objects_to_keep.append(obj)
    #     else:
    #         logger.info(f'Removed object with bbox center height: {obj["bbox"].center[2]}')

    # objects = MapObjectList(objects_to_keep)
    logger.info(f"After postprocessing: {len(objects)} objects")

    return objects

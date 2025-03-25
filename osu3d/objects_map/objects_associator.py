from osu3d.objects_map.utils import process_pcd, get_bounding_box, \
    compute_spatial_similarities, compute_visual_similarities, merge_obj2_into_obj1


class ObjectsAssociator:
    def __init__(self,
                 merge_det_obj_spatial_sim_thresh, # 空间相似度阈值
                 merge_det_obj_visual_sim_thresh, # 视觉相似度阈值
                 downsample_voxel_size, # 体素降采样的尺寸
                 **kwargs):
        self.merge_det_obj_spatial_sim_thresh = merge_det_obj_spatial_sim_thresh
        self.merge_det_obj_visual_sim_thresh = merge_det_obj_visual_sim_thresh
        self.downsample_voxel_size = downsample_voxel_size

    def __call__(self, detected_objects, scene_objects):
        # compute spatial sim 计算空间相似度
        spatial_sim = compute_spatial_similarities(detected_objects, scene_objects)
        spatial_sim[spatial_sim <= self.merge_det_obj_spatial_sim_thresh] = float('-inf')

        # compute vis sim 计算视觉相似度
        visual_sim = compute_visual_similarities(detected_objects, scene_objects, spatial_sim)
        visual_sim[visual_sim < self.merge_det_obj_visual_sim_thresh] = float('-inf')

        # merge det to objects 合并更新
        scene_objects = self.merge_detections_to_objects(detected_objects, scene_objects,
            visual_sim, self.downsample_voxel_size)

        return scene_objects

    def merge_detections_to_objects(self, detected_objects, scene_objects, visual_sim, downsample_voxel_size):
        # Iterate through all detections and merge them into objects
        for i in range(visual_sim.shape[0]):

            # If not matched to any object, add it as a new object 若object i在视觉和空间上都没有足够相似的物体，则添加为新物体
            if visual_sim[i].max() == float('-inf'):
                scene_objects.append(detected_objects[i])

            # Merge with most similar existing object 寻找最相似的物体
            else:
                j = visual_sim[i].argmax()
                matched_det = detected_objects[i]
                matched_obj = scene_objects[j]
                merged_obj = merge_obj2_into_obj1(matched_obj, matched_det,
                    downsample_voxel_size, run_dbscan=False, are_objects=False)
                scene_objects[j] = merged_obj

        return scene_objects

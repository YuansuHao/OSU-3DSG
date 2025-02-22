import cv2
import torch
import numpy as np
import open3d as o3d
from loguru import logger
import torch.nn.functional as F

from osu3d.objects_map.utils import DetectionList
from osu3d.objects_map.utils import process_pcd, get_bounding_box


def from_intrinsics_matrix(K):
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy

def to_scalar(d):
    if isinstance(d, float):
        return d
    
    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()
    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()
    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")

class DetectionsAssembler:
    '''
    将当前帧的彩色图像、深度图像、相机参数、物体掩码和 DINO 特征进行聚合，得到 detected_objects
    
    负责对每一帧中的物体进行处理，包括过滤、点云生成、噪声去除和特征提取。
    mask_conf_threshold: 0.95   只有置信度高于 95% 的object mask才会被保留
    mask_area_threshold: 500    只有mask面积大于 500 个像素的物体才会被保留
    max_bbox_area_ratio: 0.75   object的边界框面积不能超过图像面积的 75%，避免处理过大的检测结果
    min_points_threshold: 150   生成的点云必须至少包含 150 个点，否则该物体检测将被忽略
    downsample_voxel_size: 0.025    定义了体素的大小，决定了降采样的粒度。通过降采样减少点云的分辨率，以便处理更高效，同时去除过密的点
    dbscan_remove_noise: True   启用DBSCAN
    dbscan_eps: 0.05            点之间的最大距离为0.05, 距离小于这个值的点会被归为同一个聚类。
    dbscan_min_points: 7        一个聚类至少需要7个点, 少于这个点数的簇会被认为是噪声。
    image_area: 1254528 # desired_height * desired_width    图像的总像素面积为1254528像素
    '''
    def __init__(self, mask_conf_threshold,
                       mask_area_threshold,
                       max_bbox_area_ratio,
                       min_points_threshold,
                       downsample_voxel_size,
                       dbscan_remove_noise,
                       dbscan_eps,
                       dbscan_min_points,
                       image_area,
                **kwargs):
        self.mask_conf_threshold = mask_conf_threshold
        self.mask_area_threshold = mask_area_threshold
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.min_points_threshold = min_points_threshold
        self.downsample_voxel_size = downsample_voxel_size
        self.dbscan_remove_noise = dbscan_remove_noise
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.image_area = image_area

    def __call__(self, step_idx, color, depth, intrinsics, pose, masks_result, descriptors):
        """
        mask_result xyxy边界框 mask掩码 confidence置信度 
        descriptors   特征 DINOv2 输出的
        """

        detection_list = DetectionList()

        # filter low confidence proposals 过滤置信度低的检测
        idx_to_save = []                                   #####  保存符合置信度要求的mask索引
        n_masks = len(masks_result["xyxy"])              ####  几个mask   几个物体
        for mask_idx in range(n_masks):
            if masks_result['confidence'][mask_idx] < self.mask_conf_threshold:          ### 去除低于置信度的
                logger.debug(f"skipping mask with a low confidence. idx = {mask_idx}")
            else:
                idx_to_save.append(mask_idx)

        masks_result["mask"] = np.take(masks_result["mask"], idx_to_save, axis=0)           ### np.take 按照idx_to_save 提取对应索引的值
        masks_result["xyxy"] = np.take(masks_result["xyxy"], idx_to_save, axis=0)
        masks_result["confidence"] = np.take(masks_result["confidence"], idx_to_save, axis=0)
        
        # compute the containing relationship among all detections and subtract fg from bg objects 计算所有边界框的包含关系, 移除被包含的检测
        masks_result['mask'] = self.mask_subtract_contained(masks_result['xyxy'], masks_result['mask'])

        # iterate over objects 遍历过滤后的所有objects 
        n_masks = len(masks_result['xyxy'])
        for mask_idx in range(n_masks):
            mask = masks_result['mask'][mask_idx]
            if mask.sum() < max(self.mask_area_threshold, 10):        #去除太小的物体
                logger.debug(f"Skipping: mask is too small for idx = {mask_idx}")
                continue
            
            # skip small outliers 边界框面积超过了图像的max_bbox_area_ratio比例，则过滤
            x1, y1, x2, y2 = masks_result['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area > self.max_bbox_area_ratio * self.image_area:      ### 边界框的面积不能超过图像的比例
                logger.debug(f"""Skipping object with area {bbox_area} > {self.max_bbox_area_ratio} * {self.image_area}.
                             idx = {mask_idx}""")
                continue

            # create object pcd生成物体的3D点云 ——> create_object_pcd
            camera_object_pcd = self.create_object_pcd(color, mask, depth, intrinsics)           ### 构建点云对象
            if len(camera_object_pcd.points) < max(self.min_points_threshold, 5):                 #### 过滤小物体       
                logger.debug(f"""Skipping: num points {camera_object_pcd.points}
                             < min points {max(self.min_points_threshold, 5)}""")
                continue
            elif len(camera_object_pcd.points) < max(2 * self.min_points_threshold, 5):
                logger.debug(f"Warning: few points number for {mask_idx} - less than 2 * MIN_POINTS_THRESHOLD")
            global_object_pcd = camera_object_pcd.transform(pose.cpu().numpy())

            # filter noise
            # 基于体素大小（downsample_voxel_size）和 DBSCAN 聚类算法来去除噪声点云。
            global_object_pcd, perc_preserve = process_pcd(global_object_pcd, self.downsample_voxel_size,
                self.dbscan_remove_noise, self.dbscan_eps, self.dbscan_min_points, run_dbscan=True)
            if not perc_preserve:
                logger.debug(f"Skipping: dbscan find only noise for det {mask_idx}")
                continue
            elif perc_preserve <= 0.9:
                logger.debug(f"Skipping: dbscan most common cluster is not a dominant with {perc_preserve}%")
                continue

            # filter small volumes
            # 计算物体点云的边界框，检查其体积，若小于阈值1e-6，则跳过该物体
            pcd_bbox = get_bounding_box(global_object_pcd)
            pcd_bbox.color = [0, 1, 0]
            if pcd_bbox.volume() < 1e-6:
                logger.debug("Skipping: bbox volume after downsample is very low")
                continue
            
            # calculate object descriptor
            # 通过插值操作将物体的掩码调整为与 descriptors（图像特征描述符）的大小一致，并使用该掩码区域计算物体的特征描述符。
            # 最终生成的特征描述符是物体在特定区域的特征均值。
            # 获取物体的mask
            local_mask = masks_result["mask"][mask_idx]
            local_mask = torch.tensor(local_mask)
            local_mask = F.interpolate(local_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(descriptors.shape[0], descriptors.shape[1]),
                mode='nearest').squeeze(0).squeeze(0).bool()
            loc_descriptor = descriptors[local_mask].mean(dim=0, keepdim=True)
            if torch.any(torch.isnan(descriptors)):
                if not torch.any(local_mask):
                    logger.debug(f"Downsampled mask {mask_idx} does not contain any True values")
                continue
            
            # form object dict 将检测结果加入detection_list
            detected_object = {
                'pcd': global_object_pcd, # pointcloud
                'bbox': pcd_bbox, # bbox
                'descriptor': loc_descriptor, # descriptor  # [1, d]
                'num_detections': 1, # number of detections (for filtering)
                'id': {step_idx}, # detection frame idx (for projection)
            }
            detection_list.append(detected_object)

        return detection_list

    def mask_subtract_contained(self, xyxy, mask, th1=0.8, th2=0.7):
        '''
        Compute the containing relationship between all pair of bounding boxes.
        For each mask, subtract the mask of bounding boxes that are contained by it.
        处理两个边界框(bounding boxes)之间的包含关系, 并基于该关系更新物体的掩码(mask)
        Args:
            xyxy: (N, 4), in (x1, y1, x2, y2) format
            mask: (N, H, W), binary mask
            th1: float, threshold for computing intersection over box1
            th2: float, threshold for computing intersection over box2
            
        Returns:
            mask_sub: (N, H, W), binary mask
        '''

        # Get areas of each xyxy
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)            #### 计算每个边界框的面积

        # Compute intersection boxes 计算交集边界框
        lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)             #### 框之间的左上角交点
        rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)         ##   框之间右下角交点
        inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)   ####交集区域的宽高

        # Compute areas of intersection boxes 计算交集面积
        inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)          ### 交集的面积                            
        # 计算交集面积占物体面积的比例
        inter_over_box1 = inter_areas / areas[:, None] # (N, N)              计算交集面积占第一个边界框面积的比例
        # inter_over_box2 = inter_areas / areas[None, :] # (N, N)            计算交集面积占第二个边界框面积的比例
        inter_over_box2 = inter_over_box1.T # (N, N)
        # 判断包含关系
        # if the intersection area is smaller than th2 of the area of box1, 
        # and the intersection area is larger than th1 of the area of box2,
        # then box2 is considered contained by box1
        contained = (inter_over_box1 < th2) & (inter_over_box2 > th1) # (N, N)
        # 处理被包含的物体掩码
        contained_idx = contained.nonzero() # (num_contained, 2)        ## 获取所有满足包含关系的边界框对的索引
        mask_sub = mask.copy() # (N, H, W)

        # 减去被包含的物体掩码
        kernel = np.ones((5, 5), np.uint8)
        for i in range(len(contained_idx[0])):
            mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])        ## 结果是从包含边界框的掩码中减去被包含边界框的掩码。
            ### remove edge noise after substraction
            mask_ = cv2.numpy.asmatrix(mask_sub[contained_idx[0][i]].astype(np.uint8))
            mask_ = cv2.erode(mask_, kernel, iterations=2)              ### 腐蚀操作，用于去除掩码边缘的噪声
            mask_ = cv2.dilate(mask_, kernel, iterations=2)             ###  膨胀操作，用于恢复掩码的主体部分
            mask_sub[contained_idx[0][i]] = np.asarray(mask_, bool)
        return mask_sub
    
    def create_object_pcd(self, color, mask, depth_array, cam_K):
        depth_array = depth_array[..., 0].cpu().numpy()            ### 处理depth数据 
        # Also remove points with invalid depth values 过滤掉无效的深度值

        mask = np.logical_and(mask, depth_array > 0)          ### 去除深度为0的无效点

         # 检查是否有有效的点
        if mask.sum() == 0:                       ### 没有有效的 就跳出
            pcd = o3d.geometry.PointCloud()
            return pcd
        
        # 提取相机内参
        fx, fy, cx, cy = from_intrinsics_matrix(cam_K)         ### 相机参数
        # 生成图像的像素坐标网格
        height, width = depth_array.shape
        x = np.array(np.arange(0, width, 1.0))           ###   [0,1,2,....  ]
        y = np.array(np.arange(0, height, 1.0))          ########生成宽度高度的坐标

        u, v = np.meshgrid(x, y)                       ### 生成网格
        
        # Apply the mask, and unprojection is done only on the valid points 使用掩码，选取有效像素的坐标和深度值
        masked_depth = depth_array[mask] # (N, )            ### 选取有效的深度
        u = u[mask] # (N, )
        v = v[mask] # (N, )                                 ### 有效的坐标

        # Convert to 3D coordinates 将2D图像坐标转换为3D空间坐标
        x = (u - cx) * masked_depth / fx
        y = (v - cy) * masked_depth / fy
        z = masked_depth

        # Stack x, y, z coordinates into a 3D point cloud 将3D坐标堆叠为点云格式
        points = np.stack((x, y, z), axis=-1)
        points = points.reshape(-1, 3)
        
        # Perturb the points a bit to avoid colinearity 增加轻微扰动，避免共线性问题
        points += np.random.normal(0, 4e-3, points.shape)

        if points.shape[0] == 0:
            raise RuntimeError("zero points pcd")

        # Create an Open3D PointCloud object 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # colors = color.numpy()[mask] / 255.0
        obj_color = np.random.rand(3)
        colors = np.full(points.shape, obj_color)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

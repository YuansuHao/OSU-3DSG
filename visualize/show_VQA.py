import gzip
import pickle
import json
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from osu3d.objects_map.utils import MapObjectList
# xvfb-run -s "-screen 0 1280x720x24" python3 visualize/show_VQA.py

def main():
    # office0场景
    objects_path = "/data/coding/osu3d/output/scenes/replica_room0_objects.pkl.gz"
    json_path = "/data/coding/osu3d/output/scenes/replica_room0.json"
    # query_answer_path = "/data/coding/VQA/3DSG_result/task3/room0/output/Find the furthest chair from the door.json"
    query_answer_path = "/data/coding/VQA/3DSG_result/output/Replica/room0/Find the furthest armchair from the door.json"
    output_image_path = "/data/coding/osu3d/output/query_picture/Replica/Find the furthest armchair from the door.png"
    
    # 19eda6f4-55aa-29a0-8893-8eac3a4d8193场景
    # objects_path = "/data/coding/osu3d/output/scenes/3rscan_19eda6f4-55aa-29a0-8893-8eac3a4d8193.pkl.gz"
    # json_path = "/data/coding/osu3d/output/scenes/3rscan_19eda6f4-55aa-29a0-8893-8eac3a4d8193.json"
    # query_answer_path = "/data/coding/osu3d/output/query_answer/3RScan/19eda6f4-55aa-29a0-8893-8eac3a4d8193/Is there any chair in the scene.json"
    # output_image_path = "/data/coding/osu3d/output/query_picture/3RScan/19ed-chair.png"

    # 加载 .pkl.gz 文件中的对象数据
    with gzip.open(objects_path, "rb") as f:
        results = pickle.load(f)
    objects = MapObjectList()
    objects.load_serializable(results["objects"])

    # 加载 JSON 文件中的物体信息，获取 description 字段
    with open(json_path, "r") as f:
        scene_objects = json.load(f)

    # 创建无窗口的渲染器，设置分辨率
    renderer = rendering.OffscreenRenderer(1200, 680)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # 设置背景颜色为白色

    # 获取点云和包围盒数据
    pcds = objects.get_values("pcd")
    bboxes = objects.get_values("bbox")

    # 配置材质
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"  # 使用带光照的 shader

    # 配置普通边界框材质
    normal_bbox_material = rendering.MaterialRecord()
    normal_bbox_material.shader = "unlitLine"
    normal_bbox_material.line_width = 1.0  # 普通线条厚度

    # 配置黑色边界框材质（用于匹配项）
    matched_bbox_material = rendering.MaterialRecord()
    matched_bbox_material.shader = "unlitLine"
    matched_bbox_material.line_width = 3.0  # 增加线条厚度
    matched_bbox_material.base_color = [0, 0, 0, 1.0]  # 黑色

    # 定义 30 种高对比度颜色
    high_contrast_colors = [
        [1.0, 0.0, 0.0],   # 红色
        [0.0, 1.0, 0.0],   # 绿色
        [0.0, 0.0, 1.0],   # 蓝色
        [1.0, 0.5, 0.0],   # 橙色
        [0.0, 1.0, 1.0],   # 青色
        [1.0, 1.0, 0.0],   # 黄色
        [0.5, 0.5, 0.5],   # 灰色
        [1.0, 0.0, 1.0],   # 粉红色
        [0.5, 0.25, 0.0],  # 棕色
        [0.0, 0.5, 0.5],   # 深青色
        [0.75, 0.0, 0.5],  # 紫红色
        [0.25, 0.75, 0.25],# 浅绿
        [0.5, 0.5, 0.0],   # 暗黄
        [0.75, 0.25, 0.25],# 砖红
        [0.0, 0.5, 1.0],   # 蓝绿色
        [0.75, 0.75, 0.0], # 金黄
        [0.25, 0.25, 0.75],# 深蓝
        [0.25, 0.5, 0.75], # 天蓝
        [0.5, 0.75, 0.5],  # 浅绿色
        [0.75, 0.5, 0.25], # 浅棕
        [0.5, 0.25, 0.75], # 浅紫
        [0.25, 0.75, 0.5], # 浅青
        [0.25, 0.25, 0.5], # 暗蓝
        [0.5, 0.25, 0.25], # 暗红
        [0.25, 0.5, 0.25], # 暗绿
        [0.75, 0.0, 0.25], # 深粉
        [0.5, 0.0, 0.25],  # 深紫
        [0.25, 0.0, 0.5],  # 深蓝紫
        [0.0, 0.25, 0.5]   # 深蓝绿
    ]

    # 为每个唯一的 description 分配一种颜色
    description_to_color = {}
    for obj in scene_objects:
        description = obj["description"]
        if description not in description_to_color:
            color_index = len(description_to_color) % len(high_contrast_colors)
            description_to_color[description] = high_contrast_colors[color_index]

    # 加载查询答案中的目标边界框列表
    with open(query_answer_path, "r") as f:
        query_data = json.load(f)
    structured_responses = query_data["structured_response"]

    # 匹配目标边界框
    # 遍历每个查询结果的目标边界框并进行匹配
    for response in structured_responses:
        target_bbox_center = np.array(response["bbox_center"])
        target_bbox_extent = np.array(response["bbox_extent"])

        # 匹配目标边界框
        matched_bbox = None
        for i, bbox in enumerate(bboxes):
            bbox_center = bbox.get_center()
            bbox_extent = bbox.extent

            # 检查中心和尺寸是否接近
            if np.allclose(bbox_center, target_bbox_center, atol=0.1) and np.allclose(bbox_extent, target_bbox_extent, atol=0.1):
                matched_bbox = bbox
                break

        # 渲染匹配到的边界框
        if matched_bbox:
            print(f"matched_bbox_center:{matched_bbox.get_center()}")
            print(f"matched_bbox_extent:{matched_bbox.extent}")
            # 调整颜色以突出匹配的边界框
            renderer.scene.add_geometry("matched_bbox", matched_bbox, matched_bbox_material)


    # 添加所有点云和边界框到渲染器中
    for i in range(len(bboxes)):
        # 调整体素大小以增加点云密度
        pcd_i = pcds[i].voxel_down_sample(voxel_size=0.01)
        # 获取该物体的描述，并分配颜色
        description = scene_objects[i]["description"]
        color = description_to_color[description]
        pcd_i.paint_uniform_color(color)
        renderer.scene.add_geometry(f"pcd_{i}", pcd_i, material)

        # 如果是匹配的边界框，用黑色加粗框表示，否则使用普通边界框
        if bboxes[i] == matched_bbox:
            renderer.scene.add_geometry(f"matched_bbox_{i}", bboxes[i], matched_bbox_material)
        # else:
            # renderer.scene.add_geometry(f"bbox_{i}", bboxes[i], normal_bbox_material)

    # 设置相机视角，以显示整个场景
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(np.vstack([pcd.points for pcd in pcds]))
    )
    center = bounding_box.get_center()
    extent = bounding_box.get_extent()
    renderer.setup_camera(60.0, center, center + [0, 0, extent[2]*2], [0, 1, 0])

    # 渲染并保存图像
    image = renderer.render_to_image()
    o3d.io.write_image(output_image_path, image)
    print(f"绘制后的全景图已保存至: {output_image_path}")

if __name__ == "__main__":
    main()

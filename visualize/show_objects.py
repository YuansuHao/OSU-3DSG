import gzip
import pickle
import json
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from osu3d.objects_map.utils import MapObjectList
#print(o3d.__DEVICE_API__)
# xvfb-run -s "-screen 0 1280x720x24" python3 visualize/show_objects.py

def main():
    # 写死文件路径
    #objects_path = "/data/coding/osu3d/office_data/output/12.05.2024_00:58:17_office.pkl.gz"
    #json_path = "/data/coding/osu3d/office_data/output/12.05.2024_00:58:17_office.json"
    #output_image_path = "/data/coding/osu3d/office_data/output/object_map/office.png"
    objects_path = "/data/coding/datasets/room_data/office_room1/output/12.18.2024_12:37:11_office.pkl.gz"
    json_path = "/data/coding/datasets/room_data/office_room1/output/12.18.2024_12:37:11_office.json"
    output_image_path = "/data/coding/datasets/room_data/office_room1/output/office.png"
    #print(1)
    # bbox_

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
    

    # 设置背景颜色为白色
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    # 获取点云和包围盒数据
    pcds = objects.get_values("pcd")
    bboxes = objects.get_values("bbox")

    # 配置材质
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"  # 使用带光照的 shader

    # 配置边界框材质
    bbox_material = rendering.MaterialRecord()
    bbox_material.shader = "unlitLine"  # 使用未受光照影响的线条
    bbox_material.line_width = 2.0  # 增加线条厚度

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

    # 添加点云和包围盒到渲染器的场景中
    for i in range(len(bboxes)):
        # 添加边界框，使用设置好的材质
        # renderer.scene.add_geometry(f"bbox_{i}", bboxes[i], bbox_material)

        # 调整体素大小以增加点云密度
        pcd_i = pcds[i].voxel_down_sample(voxel_size=0.005)

        # 获取该物体的描述，并分配颜色
        description = scene_objects[i]["description"]
        color = description_to_color[description]
        pcd_i.paint_uniform_color(color)
        renderer.scene.add_geometry(f"pcd_{i}", pcd_i, material)

    # 获取包围整个点云的轴对齐边界框
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(np.vstack([pcd.points for pcd in pcds]))
    )
    center = bounding_box.get_center()
    extent = bounding_box.get_extent()

    # 通过设置相机的 look_at 参数来调整视角
    renderer.setup_camera(60.0, center, center + [0, 0, extent[2]*3], [0, 1, 0])

    # 渲染并保存图像
    image = renderer.render_to_image()
    o3d.io.write_image(output_image_path, image)

if __name__ == "__main__":
    main()

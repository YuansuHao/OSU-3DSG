import gzip
import pickle
import json
import numpy as np
import pyvista as pv

from osu3d.objects_map.utils import MapObjectList
#print(o3d.__DEVICE_API__)
# xvfb-run -s "-screen 0 1280x720x24" python3 visualize/show_objects.py

def main():
    # 写死文件路径
    #objects_path = "/data/coding/osu3d/office_data/output/12.05.2024_00:58:17_office.pkl.gz"
    #json_path = "/data/coding/osu3d/office_data/output/12.05.2024_00:58:17_office.json"
    #output_image_path = "/data/coding/osu3d/office_data/output/object_map/office.png"
    objects_path = "/data/coding/osu3d/output/scenes/12.18.2024_17:02:52_replica_room2_objects.pkl.gz"
    json_path = "/data/coding/osu3d/output/scenes/12.18.2024_17:02:52_replica_room2.json"
    output_image_path = "/data/coding/osu3d/output/object_map/office.png"
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

    # 获取点云和包围盒数据
    pcds = objects.get_values("pcd")
    bboxes = objects.get_values("bbox")


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

    # 启动Xvfb以支持无头环境中的渲染
    pv.start_xvfb()

    # 创建 PyVista Plotter 对象并设置为离屏模式
    plotter = pv.Plotter(off_screen=True, window_size=(2000, 1200))  # 无窗口模式
    plotter.enable_anti_aliasing('msaa')

    # 添加点云和包围盒到渲染器的场景中
    for i in range(len(pcds)):
        # 调整体素大小以增加点云密度
        pcd_i = pcds[i].voxel_down_sample(voxel_size=0.005)

        # 获取该物体的描述，并分配颜色
        description = scene_objects[i]["description"]
        color = description_to_color[description]

        # 将 Open3D 点云转换为 PyVista 点云
        points = np.asarray(pcd_i.points)
        point_cloud = pv.PolyData(points)
        point_cloud["color"] = np.tile(color, (points.shape[0], 1))  # 为每个点分配颜色

        # 添加点云到绘图器
        plotter.add_mesh(point_cloud, color=color, point_size=10,specular=0.5, specular_power=15)

    # 获取包围整个点云的轴对齐边界框
    all_points = np.vstack([np.asarray(pcd.points) for pcd in pcds])
    bounding_box = pv.PolyData(all_points).bounds
    center = [(bounding_box[0] + bounding_box[1]) / 2,
              (bounding_box[2] + bounding_box[3]) / 2,
              (bounding_box[4] + bounding_box[5]) / 2]

    # 设置相机视角
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1)  # 调整缩放

    # 渲染并保存图像
    plotter.show(screenshot=output_image_path,auto_close=False)



if __name__ == "__main__":
    main()

import os
import gzip
import pickle
import argparse

import imageio
import numpy as np
import open3d as o3d

from osu3d.datasets import get_dataset
from osu3d.objects_map.utils import MapObjectList


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def main(args):
    meta_path = os.path.join(args.animation_folder, "meta.pkl.gz")
    with gzip.open(meta_path, "rb") as file:
        meta_info = pickle.load(file)
    config = meta_info["config"]

    # Create an offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        width=config["dataset"]["desired_width"],
        height=config["dataset"]["desired_height"]
    )

    result_frames = []
    rgbd_dataset = get_dataset(config["dataset"])
    for step_idx, frame in enumerate(rgbd_dataset):
        color, _, _, pose = frame

        # Load the mapping results up to this frame
        with gzip.open(os.path.join(
            args.animation_folder, f"frame_{step_idx}_objects.pkl.gz"), "rb") as file:
                frame_objects = pickle.load(file)

        frame_objects_list = MapObjectList()
        frame_objects_list.load_serializable(frame_objects["objects"])

        # Add geometries to the renderer
        renderer.scene.clear_geometry()
        pcds = frame_objects_list.get_values("pcd")
        bboxes = frame_objects_list.get_values("bbox")
        for i, geom in enumerate(pcds + bboxes):
            renderer.scene.add_geometry(f"geom_{i}", geom, o3d.visualization.rendering.MaterialRecord())

        # Set camera pose
        center = [0, 0, 0]
        # eye = [pose[0, 3], pose[1, 3], pose[2, 3]]
        # up = [0, 1, 0]
        eye = [0, 0, 15] 
        up = [0, 0, 1] 
        renderer.scene.camera.look_at(center, eye, up)

        # Render the current frame
        render_rgb = np.asarray(renderer.render_to_image())

        # Save frames
        # image_stack = np.concatenate([color, render_rgb], axis=1)
        # result_frames.append(image_stack)

        result_frames.append(render_rgb)


    # Save the resulting video
    os.makedirs(os.path.dirname(args.video_save_path), exist_ok=True)
    imageio.mimwrite(args.video_save_path, result_frames, fps=float(args.fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--animation_folder",
                        help="folder where the objects of the mapping process are stored.")
    parser.add_argument("--fps", default=5)
    parser.add_argument("--video_save_path",
                        default="output.mp4")
    args = parser.parse_args()
    main(args)

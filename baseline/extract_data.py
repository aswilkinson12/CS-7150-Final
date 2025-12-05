import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

# settings
DATA_ROOT = "../data"                
VERSION = "v1.0-mini"            
PAST_STEPS = 4                    # 2 seconds history (4 × 0.5s)
FUTURE_STEPS = 6                  # 3 seconds future (6 × 0.5s)
TRAIN_FRACTION = 0.8              

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

# convert quaternion to yaw angle
def quaternion_yaw(q):
    q = Quaternion(q)
    return q.yaw_pitch_roll[0]  

# convert trajectory from global world coordinates to ego-centric one
def global_to_egoframe(xs, ys, yaws):
    x0, y0 = xs[0], ys[0]
    yaw0 = yaws[0]

    # translate to origin
    xs = xs - x0
    ys = ys - y0

    # rotate by -yaw0
    cos_y = np.cos(-yaw0)
    sin_y = np.sin(-yaw0)

    xr = xs * cos_y - ys * sin_y
    yr = xs * sin_y + ys * cos_y

    return xr, yr

# extract positions for one scene 
def extract_scene_trajectory(nusc, scene):
    traj = []

    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)

        # get lidar token
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd = nusc.get("sample_data", lidar_token)

        # get ego pose
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        x, y, z = ep["translation"]
        yaw = quaternion_yaw(ep["rotation"])

        traj.append((x, y, yaw))

        sample_token = sample["next"]

    traj = np.array(traj)
    return traj[:,0], traj[:,1], traj[:,2]   # xs, ys, yaws


# slice a full scene trajectory into (past, future) windows
def slice_trajectory(xs, ys):
    past_list = []
    future_list = []

    length = len(xs)
    for t in range(PAST_STEPS, length - FUTURE_STEPS):
        past = np.stack([xs[t-PAST_STEPS:t], ys[t-PAST_STEPS:t]], axis=1)
        future = np.stack([xs[t:t+FUTURE_STEPS], ys[t:t+FUTURE_STEPS]], axis=1)
        past_list.append(past)
        future_list.append(future)

    return past_list, future_list


if __name__ == "__main__":
    print("Loading nuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)

    all_past = []
    all_future = []

    scenes = nusc.scene
    num_scenes = len(scenes)
    print(f"Found {num_scenes} scenes.")

    for idx, scene in enumerate(scenes):
        print(f"\nProcessing scene {idx+1}/{num_scenes} ...")

        xs, ys, yaws = extract_scene_trajectory(nusc, scene)

        # ego-centric coordinates
        xs, ys = global_to_egoframe(xs, ys, yaws)

        # slice into windows
        past_list, future_list = slice_trajectory(xs, ys)

        all_past.extend(past_list)
        all_future.extend(future_list)

    all_past = np.array(all_past)
    all_future = np.array(all_future)

    print("\nTotal samples extracted:", len(all_past))
    print("Past shape:", all_past.shape)
    print("Future shape:", all_future.shape)

    # split into train and val
    total = len(all_past)
    train_size = int(TRAIN_FRACTION * total)

    train_past = all_past[:train_size]
    train_future = all_future[:train_size]

    val_past = all_past[train_size:]
    val_future = all_future[train_size:]

    # save arrays
    np.save(os.path.join(SAVE_DIR, "train_past.npy"), train_past)
    np.save(os.path.join(SAVE_DIR, "train_future.npy"), train_future)
    np.save(os.path.join(SAVE_DIR, "val_past.npy"), val_past)
    np.save(os.path.join(SAVE_DIR, "val_future.npy"), val_future)

    print("\nSaved:")
    print(" - train_past.npy")
    print(" - train_future.npy")
    print(" - val_past.npy")
    print(" - val_future.npy")
    print("\nDone!")

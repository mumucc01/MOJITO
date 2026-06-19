import lzma
import pickle
import numpy as np

metric_cache_path = "/mnt/volumes/base-3da-ali-sh-mix/mayongjia/mayongjia/mayongjia/code/world_action_model/data/navsim/download/cache/navtrain/2021.10.11.08.31.07_veh-50_00282_00680/unknown/aea5e098122c5c2b/metric_cache.pkl"

with lzma.open(metric_cache_path, "rb") as f:
    metric_cache = pickle.load(f)

traj = metric_cache.trajectory  # InterpolatedTrajectory
states = traj._trajectory       # List[EgoState], len=51

arr = np.array(
    [[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading, s.time_point.time_s] for s in states],
    dtype=np.float64,
)

print("type(traj):", type(traj))
print("num_states:", len(states))
print("arr.shape:", arr.shape)          # (51, 4)
print("columns: [x, y, heading, time_s]")

np.set_printoptions(precision=4, suppress=True)
print("\nfirst 5:\n", arr[:5])
print("\nlast 5:\n", arr[-5:])

poses = arr[:, :3]
print("\nposes.shape:", poses.shape)    # (51, 3)

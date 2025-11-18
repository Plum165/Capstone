from helpers import *
from pathlib import Path
import numpy as np

# ---  Initialize system with config ---
init("config.yml", is_for_photo=True)
config = shared['config']

# ---  Initialize renderer ---
init_renderer(scale=0.5, load_mesh=True)

# ---  Load camera parameters from waypoints folder ---
json_path = Path("waypoints") / config['scene_name'] / "cameras_unconstrained.json"
if not json_path.exists():
    raise FileNotFoundError(f"Camera JSON file not found at {json_path}")
params = load_camera_parameters(str(json_path))

print(f"Number of waypoints: {params['num_cameras']}")
print(f"Has route information: {params['has_route']}")
print(f"Waypoint data includes:")
print(f"  - Positions shape: {params['positions'].shape}")
print(f"  - Rotations shape: {params['rotations'].shape}")
if params['has_route']:
    print(f"  - Route order: {len(params['waypoint_order'])} waypoints")

# ---  Choose the waypoint to render ---
ci = 12  # change to whichever waypoint you want
origin = params['positions'][ci]
yaw = params['rotations'][ci, 0]
pitch = params['rotations'][ci, 1]

# --- Render scene from this UAV viewpoint ---
colormap, depthmap = render_scene(origin, yaw, pitch)

#  Save or display the image ---
# Display in notebook or interactive session
show2d([colormap], dpi=100)

# Optional: save to file
from imageio import imwrite
imwrite("uav_view.png", colormap)
print("Rendered UAV perspective saved as uav_view.png")

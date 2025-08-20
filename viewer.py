# viewer.py
import json
import numpy as np
from helpers import (
    load_camera_parameters,
    generate_camera_3d,
    create_route_trace,
    show3d_plotly,
)
import plotly.io as pio

pio.renderers.default = "browser"  # Opens in your default web browser

class Viewer:
    """
    Wrapper around helpers.py for loading JSON camera paths
    and rendering them interactively with Plotly.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.params = load_camera_parameters(json_path)
        self.traces = []

    def build_scene(self):
        """Generate camera objects and route (if available)."""
        self.traces = []
        for idx in range(self.params["num_cameras"]):
            origin = self.params["positions"][idx]
            yaw, pitch = self.params["rotations"][idx]
            self.traces.extend(
                generate_camera_3d(origin, yaw, pitch, text=f"Cam {idx+1}")
            )

        # Add route if it exists
        route_trace = create_route_trace(self.params)
        if route_trace:
            self.traces.append(route_trace)

    def show(self):
        """Display the 3D scene in Plotly."""
        if not self.traces:
            self.build_scene()
        show3d_plotly(self.traces)


if __name__ == "__main__":
    # Example usage:
    viewer = Viewer("cameras_layer.json")  # replace with your JSON path for now
    viewer.build_scene()
    viewer.show()

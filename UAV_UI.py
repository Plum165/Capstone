import sys
import json
import tempfile
from pathlib import Path
import traceback

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly.io as pio

from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QButtonGroup, QTableWidgetItem, QMessageBox, QVBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
from PyQt6.QtWebChannel import QWebChannel

import calculations_suite
# helpers should provide: init, init_renderer, shared, load_camera_parameters,
# generate_camera_3d_thickness, render_scene, show2d, show3d_plotly
from helpers import (
    init, init_renderer, shared, load_camera_parameters,
    generate_camera_3d_thickness, render_scene, show2d, show3d_plotly
)


class MainWorkingWindow(QtWidgets.QMainWindow):
    '''
    Main UAV visualisation window.
    '''
    def __init__(self):
        super().__init__()
        uic.loadUi("UI/UAV-mainWorkingWindow.ui", self)
        print("UI file 'UAV-mainWorkingWindow.ui' loaded.")

        # --- State Initialization ---
        self.camera_parameters = None
        self.drone_pos = np.array([0.0, 0.0, 5.0], dtype=float)
        self.drone_yaw = 0.0
        self.drone_pitch = 0.0
        
        # --- UI Initialization ---
        # Find the placeholder for the plot
        placeholder = self.findChild(QtWidgets.QWidget, "plotPlaceholder")
        if placeholder:
            self.plotlyWidget = QWebEngineView()
            layout = QVBoxLayout(placeholder)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.plotlyWidget)
            print("Plot widget successfully placed in 'plotPlaceholder'.")
        else:
            print("FATAL ERROR: Could not find a QWidget named 'plotPlaceholder' in UI file.")
            QMessageBox.critical(self, "UI Error", "Could not find a widget named 'plotPlaceholder'. Please add one in Qt Designer.")
            return

        self.init_controlPanel()
        self.populate_model_list()

        # Connect FPV control buttons
        self._connect_buttons()

    def _connect_buttons(self):
        """Connects the drone and camera control buttons."""
        print("Connecting drone control buttons...")
        bindings = {
            "forwardButton": lambda: self.move_drone("fwd"),
            "backwardButton": lambda: self.move_drone("back"),
            "leftButton": lambda: self.move_drone("left"),
            "rightButton": lambda: self.move_drone("right"),
            "uButton": lambda: self.altitude("up"),
            "dButton": lambda: self.altitude("down"),
            "tupButton": lambda: self.tilt_cam("up"),
            "tdownButton": lambda: self.tilt_cam("down"),
            "tleftButton": lambda: self.yaw_cam("left"),
            "trightButton": lambda: self.yaw_cam("right"),
        }
        for name, func in bindings.items():
            button = self.findChild(QtWidgets.QPushButton, name)
            if button:
                button.clicked.connect(func)
            else:
                print(f"  - Warning: Button '{name}' not found in UI file.")
        print("Button connections complete.")

    def init_selectModelTab(self):
        selectModelTab = self.controlPanel.widget(0)
        self.refreshModelListButton = selectModelTab.findChild(QtWidgets.QPushButton, "refreshModelListButton")
        self.loadModelButton = selectModelTab.findChild(QtWidgets.QPushButton, "loadModelButton")
        self.closeModelButton = selectModelTab.findChild(QtWidgets.QPushButton, "closeModelButton")
        self.showWaypointsCheckBox = selectModelTab.findChild(QtWidgets.QCheckBox, "showWaypointsCheckBox")
        self.showSplineCheckBox = selectModelTab.findChild(QtWidgets.QCheckBox, "showSplineCheckBox")
        self.layerRadioButton = selectModelTab.findChild(QtWidgets.QRadioButton, "layerRadioButton")
        self.unconstrainedRadioButton = selectModelTab.findChild(QtWidgets.QRadioButton, "unconstrainedRadioButton")
        self.routeRadioButtonGroup = QButtonGroup(self)
        self.routeRadioButtonGroup.addButton(self.layerRadioButton)
        self.routeRadioButtonGroup.addButton(self.unconstrainedRadioButton)
        self.routeFrame = selectModelTab.findChild(QtWidgets.QFrame, "routeFrame")
        
        self.closeModelButton.setEnabled(False)

        self.refreshModelListButton.clicked.connect(self.populate_model_list)
        self.loadModelButton.clicked.connect(self.load_model)
        self.closeModelButton.clicked.connect(self.close_model)

    def init_viewWaypointTab(self):
        viewWaypointTab = self.controlPanel.widget(1)
        self.closeWaypointButton = viewWaypointTab.findChild(QtWidgets.QPushButton, "closeWaypointButton")
        self.numberOfWaypointsLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "numberOfWaypointsLabel")
        self.selectWaypointSpinBox = viewWaypointTab.findChild(QtWidgets.QSpinBox, "selectWaypointSpinBox")
        self.waypointsRangeLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "waypointsRangeLabel")
        self.viewWaypointButton = viewWaypointTab.findChild(QtWidgets.QPushButton, "viewWaypointButton")
        self.waypointInformationLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "waypointInformationLabel")
        self.outputFrame = viewWaypointTab.findChild(QtWidgets.QFrame, "outputFrame")
        self.waypointXOutputLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "waypointXOutputLabel")
        self.waypointYOutputLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "waypointYOutputLabel")
        self.waypointZOutputLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "waypointZOutputLabel")
        self.waypointYawOutputLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "waypointYawOutputLabel")
        self.waypointPitchOutputLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "waypointPitchOutputLabel")
        
        # Check for the selectWaypointsLabel, it might not exist in all UI versions
        self.selectWaypointsLabel = viewWaypointTab.findChild(QtWidgets.QLabel, "selectWaypointsLabel")

        self.closeWaypointButton.setEnabled(False)
        self.controlPanel.setTabEnabled(1, False)
        self.outputFrame.hide()

        self.closeWaypointButton.clicked.connect(self.close_waypoint)
        self.viewWaypointButton.clicked.connect(self.view_waypoint)

    def init_calculationInfoTab(self):
        calculationInfoTab = self.controlPanel.widget(2)
        # Add your calculationInfoTab initialization code here if needed

    def init_fpvTab(self):
        fpvTab = self.controlPanel.widget(3)
        self.controlPanel.setTabEnabled(3, False)

    def init_controlPanel(self):
        self.controlPanel.setCurrentIndex(0)
        self.init_selectModelTab()
        self.init_viewWaypointTab()
        self.init_calculationInfoTab()
        self.init_fpvTab()

    def populate_model_list(self):
        print("Loading/refreshing model list")
        self.selectModelList.clear()
        folder_names = calculations_suite.get_folder_names_in("./models")
        for name in folder_names:
            self.selectModelList.addItem(name)
        self.selectModelList.clearSelection()
        print("Model list is now up to date")

    def load_model(self):
        selected_model_items = self.selectModelList.selectedItems()
        selected_waypoint_route_type = self.routeRadioButtonGroup.checkedButton()

        if not selected_model_items or not selected_waypoint_route_type:
            QMessageBox.warning(self, "Model loading requirements not met", "Please select a model and a path type.")
            return

        try:
            selected_model = selected_model_items[0].text()
            print(f"Opening the model: {selected_model}")

            self.viewportLabel.setText(f"{selected_model} [{selected_waypoint_route_type.text()} trajectory]")

            init("config.yml")
            shared["config"]["scene_name"] = selected_model
            config = shared["config"]
            init_renderer(scale=0.5, load_mesh=True)

            dataset_path = Path(config['dataset_path'])
            mesh_path = dataset_path / selected_model / "MeshLR" / f"{selected_model}.ply"
            msh = o3d.io.read_triangle_mesh(str(mesh_path))
            simplified_mesh = msh.simplify_vertex_clustering(voxel_size=0.2, contraction=o3d.geometry.SimplificationContraction.Average)

            plotly_objs = []
            vertices = np.asarray(simplified_mesh.vertices)
            
            if simplified_mesh.has_vertex_colors():
                v_colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in simplified_mesh.vertex_colors]
                mesh_trace = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=np.asarray(simplified_mesh.triangles)[:,0], j=np.asarray(simplified_mesh.triangles)[:,1], k=np.asarray(simplified_mesh.triangles)[:,2], vertexcolor=v_colors, name='3D Mesh', hoverinfo='skip')
            else:
                mesh_trace = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=np.asarray(simplified_mesh.triangles)[:,0], j=np.asarray(simplified_mesh.triangles)[:,1], k=np.asarray(simplified_mesh.triangles)[:,2], color="lightgrey", name='3D Mesh', hoverinfo='skip')
            plotly_objs.append(mesh_trace)

            cameras_file = f"cameras_{selected_waypoint_route_type.text()}.json"
            json_path = Path("waypoints") / selected_model / cameras_file
            if json_path.exists():
                params = load_camera_parameters(json_path)
                self.camera_parameters = params
                positions, waypoint_order = params["positions"], params["waypoint_order"]
                self.drone_pos = np.array(positions[0])
                self.drone_yaw = float(params["rotations"][0, 0])
                self.drone_pitch = float(params["rotations"][0, 1])

                if self.showWaypointsCheckBox.isChecked():
                    ordered_waypoints = np.array([positions[i] for i in waypoint_order])
                    path_trace = go.Scatter3d(x=ordered_waypoints[:,0], y=ordered_waypoints[:,1], z=ordered_waypoints[:,2], mode="lines+markers", line=dict(color='green', width=4), name="Waypoint Path")
                    plotly_objs.append(path_trace)
            else:
                self.camera_parameters = None
                self.drone_pos = vertices.mean(axis=0)

            # Add drone visuals
            drone_marker = go.Scatter3d(x=[self.drone_pos[0]], y=[self.drone_pos[1]], z=[self.drone_pos[2]], mode="markers", marker=dict(size=8, color="red", symbol="diamond"), name="Drone")
            plotly_objs.append(drone_marker)

            fig = show3d_plotly(plotly_objs, ret=True)

            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".html", encoding="utf-8") as tmp_html_file:
                pio.write_html(fig, file=tmp_html_file.name, auto_open=False, include_plotlyjs=True)
                self.plotlyWidget.load(QUrl.fromLocalFile(str(Path(tmp_html_file.name).resolve())))
            
            self.closeModelButton.setEnabled(True)
            self.loadModelButton.setEnabled(False)
            self.selectModelList.setEnabled(False)
            self.controlPanel.setTabEnabled(1, True)
            self.controlPanel.setTabEnabled(3, True) # Enable FPV Tab
            
            if self.camera_parameters:
                num_waypoints = self.camera_parameters["num_cameras"]
                self.numberOfWaypointsLabel.setText(f"Number of available waypoints: {num_waypoints}")
                self.waypointsRangeLabel.setText(f"Range [0-{num_waypoints-1}]")
                self.selectWaypointSpinBox.setMaximum(num_waypoints-1)
            
            self.update_fpv_for_drone() # Show initial drone FPV

        except Exception:
            QMessageBox.critical(self, "Error", "An error occurred while loading the model.")
            traceback.print_exc()

    def close_model(self):
        selected_model = self.selectModelList.currentItem()
        if selected_model:
            print(f"Closing the model: {selected_model.text()}")
            self.viewportLabel.setText("No model loaded")
            self.closeModelButton.setEnabled(False)
            self.loadModelButton.setEnabled(True)
            self.selectModelList.setEnabled(True)
            self.controlPanel.setTabEnabled(1, False)
            self.controlPanel.setTabEnabled(3, False)
            self.selectModelList.clearSelection()
            self.close_waypoint()
            self.plotlyWidget.setHtml("") # Clear the view

    def view_waypoint(self):
        if not self.camera_parameters:
            QMessageBox.warning(self, "No Waypoints", "Load a model with a waypoint path first.")
            return

        waypoint_num = self.selectWaypointSpinBox.value()
        params = self.camera_parameters
        
        print(f"Viewing waypoint {waypoint_num}")
        origin = params['positions'][waypoint_num]
        yaw, pitch = params['rotations'][waypoint_num, 0], params['rotations'][waypoint_num, 1]

        colormap, _ = render_scene(origin, yaw, pitch)
        img = show2d([colormap], dpi=100, return_image=True)
        
        img_contig = np.ascontiguousarray(img)
        h, w, c = img_contig.shape
        qimage = QImage(img_contig.data, w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.waypointXOutputLabel.setText(f"{origin[0]:.2f}")
        self.waypointYOutputLabel.setText(f"{origin[1]:.2f}")
        self.waypointZOutputLabel.setText(f"{origin[2]:.2f}m")
        self.waypointYawOutputLabel.setText(f"{yaw}°")
        self.waypointPitchOutputLabel.setText(f"{pitch}°")
        self.outputFrame.show()

        self.uavImageLabel.setPixmap(pixmap)
        self.uavImageLabel.setScaledContents(True)
        self.closeWaypointButton.setEnabled(True)

    def close_waypoint(self):
        self.selectWaypointSpinBox.setValue(0)
        self.uavImageLabel.setPixmap(QPixmap("UI/waypointview.jpg"))
        self.closeWaypointButton.setEnabled(False)
        self.outputFrame.hide()

    def move_drone(self, direction):
        step = 1.0
        rad = np.radians(self.drone_yaw)
        if direction == "fwd":
            self.drone_pos[0] += step * np.cos(rad - np.pi / 2)
            self.drone_pos[1] += step * np.sin(rad - np.pi / 2)
        elif direction == "back":
            self.drone_pos[0] -= step * np.cos(rad - np.pi / 2)
            self.drone_pos[1] -= step * np.sin(rad - np.pi / 2)
        elif direction == "left":
            self.drone_pos[0] -= step * np.cos(rad)
            self.drone_pos[1] -= step * np.sin(rad)
        elif direction == "right":
            self.drone_pos[0] += step * np.cos(rad)
            self.drone_pos[1] += step * np.sin(rad)
        self._update_drone_state()

    def altitude(self, direction):
        step = 1.0
        if direction == "up":
            self.drone_pos[2] += step
        elif direction == "down":
            self.drone_pos[2] = max(0.0, self.drone_pos[2] - step)
        self._update_drone_state()

    def tilt_cam(self, direction):
        if direction == "up":
            self.drone_pitch = max(-89, self.drone_pitch - 5)
        elif direction == "down":
            self.drone_pitch = min(89, self.drone_pitch + 5)
        self._update_drone_state()

    def yaw_cam(self, direction):
        if direction == "left":
            self.drone_yaw = (self.drone_yaw - 5) % 360
        elif direction == "right":
            self.drone_yaw = (self.drone_yaw + 5) % 360
        self._update_drone_state()

    def _update_drone_state(self):
        # This function updates the drone's visuals and FPV
        self.update_fpv_for_drone()
        
        # Update the 3D plot (by regenerating it)
        # Note: This is inefficient but works for a simpler implementation.
        # It requires re-creating all plotly objects and reloading the HTML.
        # A more advanced version would use JavaScript to update just the drone marker.
        
        # To keep it simple and avoid JS, we just show the FPV changing.
        # To update the 3D marker, we would need to call parts of load_model() again.
        print(f"Drone moved to: {self.drone_pos}, Yaw: {self.drone_yaw}, Pitch: {self.drone_pitch}")


    def update_fpv_for_drone(self):
        try:
            colormap, _ = render_scene(self.drone_pos, self.drone_yaw, self.drone_pitch)
            img = show2d([colormap], dpi=100, return_image=True)
            img_contig = np.ascontiguousarray(img[:, :, :3])
            h, w, c = img_contig.shape
            qimage = QImage(img_contig.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimage)
            
            # Assuming you have a label named 'fpvImageLabel' in your FPV tab
            fpv_label = self.findChild(QtWidgets.QLabel, "fpvImageLabel")
            if fpv_label:
                fpv_label.setPixmap(pix.scaled(fpv_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            else:
                 # Fallback to the waypoint image label if the dedicated FPV one doesn't exist
                if hasattr(self, "uavImageLabel"):
                    self.uavImageLabel.setPixmap(pix.scaled(self.uavImageLabel.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

        except Exception as e:
            print(f"Error updating FPV: {e}")
            traceback.print_exc()

class logonWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("UI/UAV-logonWindow.ui", self)
        self.enterButton.clicked.connect(self.open_main_working_window)

    def open_main_working_window(self):
        self.main_working_window = MainWorkingWindow()
        self.main_working_window.show()
        self.close()

class startWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("UI/UAV-startWindow.ui", self)
        self.logOnButton.clicked.connect(self.open_logon_window)

    def open_logon_window(self):
        self.logonWindow = logonWindow()
        self.logonWindow.show()
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = startWindow()
    window.show()
    sys.exit(app.exec())
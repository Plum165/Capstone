from helpers import *
import math
from scipy import interpolate

# FORMATS AND GENERATORS
def format_spline_pts(spline_x_pts, spline_y_pts, spline_z_pts):
    '''
    Puts the spline points into the the format np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ...)

    Parameters
    ----------
    spline_x_pts : numpy.array
    spline_y_pts : numpy.array
    spline_z_pts : numpy.array
    
    Returns
    ----------
    spline_pts_in_xyz_format : numpy.array
    '''
    return np.array([spline_x_pts, spline_y_pts, spline_z_pts]).T

def remove_duplicate_waypoint_positions(positions, waypoint_order):
    '''
    Looks at the 'position' data of waypoints and removes duplicate waypoints. The yaw and pitch might be different, 
    but the XYZ of two waypoints might be the same (drone might take more than 1 image at a waypoint) along the path. If we want to generate
    a spline, then we must remove the duplicate waypoints. The order of waypoints is kept as specified by the waypoint_order parameter.
    
    Parameters
    ----------
    positions : numpy.array
        An array of (x, y, z) coordinates for all the waypoints. 
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).
    waypoint_order : numpy.array
        An array of indices that specifies the order in which waypoints are visited.

    Returns
    ----------
    cleaned_waypoint_positions : numpy.array
        The input waypoint positions but without any duplicates.
    '''

    # Could have used numpy's cool fancy indexing, positions_in_route_order = positions[waypoint_order].
    # Still preferred to use regular accessing. The above creates a copy of positions which might
    # become expensive if positions is very large.

    waypoints_without_duplicate_xyz_positions = [positions[waypoint_order[0]]]
    last_added = waypoints_without_duplicate_xyz_positions[0]

    for i in range(1, len(waypoint_order)):
        current_waypoint_index = waypoint_order[i]
        current_waypoint = positions[current_waypoint_index]

        # Can improve on this logic later, if there are any floating point issues.
        # It works for now. 
        if not (current_waypoint[0]==last_added[0] and
            current_waypoint[1]==last_added[1] and
            current_waypoint[2]==last_added[2]):
            waypoints_without_duplicate_xyz_positions.append(current_waypoint)
            last_added = current_waypoint

    return np.array(waypoints_without_duplicate_xyz_positions)

def generate_spline_points(positions, waypoint_order, number_of_fine_points=1000, s=0):
    '''
    Generates a spline curve from a set of (x,y,z) points.

    Parameters
    ----------
    positions : numpy.array
        An array of (x, y, z) coordinates for all the waypoints. 
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).
    waypoint_order : numpy.array
        An array of indices that specifies the order in which waypoints are visited.
    number_of_fine_points : int
        The number of points to fit on the curve to make it more or less smooth. More points means a smoother curve.
    s : int or float
        Smoothing factor that determines how closely the spine curve follows our (x,y,z) points. (s=0 will pass through every given point)
        
    Returns
    ----------
        (spline_x_pts,spline_y_pts,spline_z_pts) : tuple
         The points of the spline curve.
    '''

    # Prepare the data. Spline does not like when there are duplicate waypoints.
    # Every waypoint should have a unique (x,y,z).
    waypoints_with_unique_positions = remove_duplicate_waypoint_positions(positions, waypoint_order)

    x_pts = waypoints_with_unique_positions[:, 0]
    y_pts = waypoints_with_unique_positions[:, 1]
    z_pts = waypoints_with_unique_positions[:, 2]
    # Prepare the spline curve and get a blueprint (tck).
    # u is the % of each point along the curve.
    tck, u = interpolate.splprep([x_pts, y_pts, z_pts], s=s)

    # Generate a high-resolution set of points for the entire curve
    u_fine = np.linspace(u.min(), u.max(), number_of_fine_points)
    spline_x_fine, spline_y_fine, spline_z_fine = interpolate.splev(u_fine, tck)

    return (spline_x_fine, spline_y_fine, spline_z_fine)

def get_folder_names_in(folder_name):
	"""
	Returns all the folder names in a specific folder.
	Used to get the different models that the user can load in the ./models folder.

	Example
	----------
	```
	.
	└── models/
		├── Karner
		├── Kiri_Vehera_SriLanka
		├── Sydney_Opera_House
		├── Ziegelpfeiler
		└── User_custom_folder/
			└── Another_custom_model
	```

	The method will only return the first 4. 'Another_custom_model' will not be picked up.
	User must place models directly inside ./models.

	Parameters
	----------
	folder_name : str
		Name of folder to look in.

	Returns
	----------
	list
		Folder names inside the specified folder.
	"""
	folder_names = []
	main_folder_path = Path(folder_name)
	for element in main_folder_path.iterdir():
		if (element.is_dir()):
			folder_names.append(element.name)

	return folder_names

# HELPERS
def helper_calculate_total_distance(waypoints_in_waypoint_order):
    '''
    A helper function that handles the core logic of going over all the given waypoints
    to calculate the total path distance.

    Parameters
    ----------
    waypoints_in_waypoint_order
        An array of (x, y, z) coordinates for all the waypoints in a specified order.
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).

    Returns
    ----------
        total_distance : int or float
        The total distance of a route.
    '''
    total_path_distance = 0
    for i in range(len(waypoints_in_waypoint_order)-1):
        current_waypoint = waypoints_in_waypoint_order[i]
        next_waypoint = waypoints_in_waypoint_order[i+1]
        d = calculate_distance_between(current_waypoint, next_waypoint)
        total_path_distance += d
    
    return total_path_distance

def helper_calculate_total_vertical_distance(waypoints_in_waypoint_order):
    '''
    A helper function that handles the core logic of going over all the given waypoints
    to calculate the total vertical distance.

    Parameters
    ----------
    waypoints_in_waypoint_order
        An array of (x, y, z) coordinates for all the waypoints in a specified order.
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).

    Returns
    ----------
        total_distance : int or float
        The total vertican distance travelled in a route.
    '''
    total_vertical_distance = 0
    for i in range(len(waypoints_in_waypoint_order)-1):
        current_point = waypoints_in_waypoint_order[i]
        next_point = waypoints_in_waypoint_order[i+1]
        h = calculate_vertical_distance_between(current_point, next_point)
        total_vertical_distance += h

    return total_vertical_distance

def helper_calculate_total_rotation_angle(waypoints_in_waypoint_order, count_number_of_sharp_corners=False):
    '''
    A helper function that handles the core logic of going over all the given waypoints
    to calculate the total path rotation angle.

    Parameters
    ----------
    waypoints_in_waypoint_order
        An array of (x, y, z) coordinates for all the waypoints in a specified order.
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).
    (optional) count_number_of_sharp_corners : boolean
        A boolean value that determines if the number of sharp corners (angle > 90 degrees), in the flight path, should be counted.
        The default value is False.
        
    Returns
    ----------
    cumulative_rotation_angle : int or float
        The cumulative path rotation angle (in radians).
    (optional) number_of_sharp_corners : int
        A boolean value that determines if the number of sharp corners (angle > 90 degrees), in the flight path, should be counted.
    '''
    cumulatitve_path_rotation_angle = 0
    sharp_corner_threshold = math.pi / 2 # 90 degrees
    number_of_sharp_corners = 0
    for i in range(len(waypoints_in_waypoint_order)-2):
        current_position = waypoints_in_waypoint_order[i]
        next_position = waypoints_in_waypoint_order[i+1]
        position_after_next = waypoints_in_waypoint_order[i+2]

        vector_1 = calculate_vector_between(current_position, next_position)
        vector_2 = calculate_vector_between(next_position, position_after_next)
        angle = calculate_rotation_angle_between(vector_1, vector_2)
        if (count_number_of_sharp_corners and angle > sharp_corner_threshold):
            number_of_sharp_corners += 1
        cumulatitve_path_rotation_angle += angle

    if (count_number_of_sharp_corners):
        return cumulatitve_path_rotation_angle, number_of_sharp_corners
    
    return cumulatitve_path_rotation_angle

def helper_calculate_flight_duration(waypoint_order, 
                                total_path_distance, 
                                flight_time_to_first_waypoint_s=0, 
                                cruising_speed_ms=5, 
                                hover_time_at_waypoint=3, 
                                flight_time_back_to_base=0):
    '''
    Parameters
    ----------
    waypoint_order : numpy.array
        An array of indices that specifies the order in which waypoints are visited.
    total_path_distance : int or float
        The total distance of a route.
    (optional) flight_time_to_first_waypoint_s
        The total time in seconds it would take for the drone to takeoff and fly to the first waypoint in the route.
        The default is 0s.
    (optional) cruising_speed_ms
        The cruising speed of the drone in m/s. 
        The default is 5m/s
    (optional) hover_time_at_waypoint
        The amount of time the drone hovers at a waypoint in seconds.
        The default is 3s.
    (optional) flight_time_back_to_base
        The total time in seconds it would take for the drone to fly from the last waypoint back to where the operator tells it to go (usually base).
        The default is 0s.
    
    Returns
    ----------
    time : int or float
        The estimated flight duration time in seconds.
    '''
    # Method 1
    # total_flight_time = 0
    # for i in range(len(waypoint_order)-1):
    #     current_waypoint_index = waypoint_order[i]
    #     next_waypoint_index = waypoint_order[i+1]
    #     d = calculate_distance_between(positions[current_waypoint_index], positions[next_waypoint_index])
    #     flight_time_to_next_waypoint = d / cruising_speed_ms
    #     #print(f"{current_waypoint_index} to {next_waypoint_index} with d={d}, t={flight_time_to_next_waypoint}")
    #     total_flight_time += (hover_time_at_waypoint + flight_time_to_next_waypoint)
    # # Remember the last waypoint's hover time.
    # total_flight_time += 3

    # Method 2
    # Vertex
    number_of_waypoints = len(waypoint_order)
    total_path_flight_time = total_path_distance / cruising_speed_ms
    total_flight_time = number_of_waypoints * hover_time_at_waypoint + total_path_flight_time

    return flight_time_to_first_waypoint_s + total_flight_time + flight_time_back_to_base
    
# BETWEEN
def calculate_distance_between(position_point_1_3D, position_point_2_3D):
    '''
    Calculates the straight line distance between two points in 3D space.

    Parameters
    ----------
    position_point_1_3D : numpy.array
    position_point_2_3D : numpy.array
        A position will have the form np.array([x, y, z]). 
        A point can be retrieved from the 'position' data in params.
    
    Returns
    ----------
    distance : int or float
        The straight line distance between two points.
    '''
    # Point 1
    x_1 = position_point_1_3D[0]
    y_1 = position_point_1_3D[1]
    z_1 = position_point_1_3D[2]

    # Point 2
    x_2 = position_point_2_3D[0]
    y_2 = position_point_2_3D[1]
    z_2 = position_point_2_3D[2]

    return math.sqrt((x_2-x_1)**2 + (y_2-y_1)**2 + (z_2-z_1)**2)

def calculate_vertical_distance_between(position_point_1_3D, position_point_2_3D):
    '''
    Calculates the vertical distance between two points in 3D space.

    Parameters
    ----------
    position_point_1_3D : numpy.array
    position_point_2_3D : numpy.array
        A position will have the form np.array([x, y, z]). 
        A point can be retrieved from the 'position' data in params.
    
    Returns
    ----------
    distance : int or float
        The vertical distance between two points.
    '''
    z_1 = position_point_1_3D[2]
    z_2 = position_point_2_3D[2]
    return abs(z_2 - z_1)

def calculate_vector_between(position_point_1_3D, position_point_2_3D):
    '''
    Calculates a vector between two points in 3D space.

    Parameters
    ----------
    position_point_1_3D : numpy.array
    position_point_2_3D : numpy.array
        A position will have the form np.array([x, y, z]). 
        A point can be retrieved from the 'position' data in params.
    
    Returns
    ----------
    vector : numpy.array
        The vector between 2 points in 3D space.
    '''   
    x_1 = position_point_1_3D[0]
    y_1 = position_point_1_3D[1]
    z_1 = position_point_1_3D[2]

    x_2 = position_point_2_3D[0]
    y_2 = position_point_2_3D[1]
    z_2 = position_point_2_3D[2]

    return np.array([x_2-x_1, y_2-y_1, z_2-z_1])

def calculate_rotation_angle_between(vector_1, vector_2):
    '''
    Calculates the turning angle between two waypoints

    Parameters
    ----------
    vector_1 : numpy.array
    vector_2 : numpy.array
    
    Returns
    ----------
    rotation angle : int or float
        The angle between two vectors (in radians).
    '''
    if (np.array_equal(vector_1, vector_2) or (not np.any(vector_1)) or ((not np.any(vector_2)))):
        return 0

    norm_of_vector_1 = np.linalg.norm(vector_1) 
    norm_of_vector_2 = np.linalg.norm(vector_2)
    arccos_input = np.dot(vector_1, vector_2) / (norm_of_vector_1 * norm_of_vector_2)

    # Domain of arccos is [-1, 1].
    if (arccos_input > 1):
        arccos_input = 1
    elif (arccos_input < -1):
        arccos_input = -1

    angle_between_vectors = np.arccos(arccos_input)

    return angle_between_vectors

# TOTALS
def calculate_total_path_distance(positions, waypoint_order):
    '''
    Calculates the total distance of a path defined by an ordered sequence of waypoints. Do not use this
    method for spline points. Use the dedicated spline method instead.

    Parameters
    ----------
    positions : numpy.array
        An array of (x, y, z) coordinates for all the waypoints. 
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).
    waypoint_order : numpy.array
        An array of indices that specifies the order in which waypoints are visited.
    
    Returns
    ----------
    total_distance : int or float
        The total distance of a route.
    '''
    # Could have used numpy's cool fancy indexing, positions_in_route_order = positions[waypoint_order].
    # Still preferred to use regular accessing. The above creates a copy of positions which might
    # become expensive if positions is very large.
    waypoints_in_waypoint_order = [positions[i] for i in waypoint_order]
    total_path_distance = helper_calculate_total_distance(waypoints_in_waypoint_order)
    
    return total_path_distance

def calculate_total_spline_distance(spline_x_pts, spline_y_pts, spline_z_pts):
    '''
    Calculates the total path distance of waypoints in a route.

    Parameters
    ----------
    spline_x_pts : numpy.array
        The x-coordinates for the spline curve.
    spline_y_pts : numpy.array
        The y-coordinates for the spline curve.
    spline_z_pts : numpy.array
        The z-coordinates for the spline curve.
    
    Returns
    ----------
    distance : int or float
        The total distance of the spline path ('curve')
    '''
    waypoints_in_waypoint_order = format_spline_pts(spline_x_pts, spline_y_pts, spline_z_pts)
    total_spline_path_distance = helper_calculate_total_distance(waypoints_in_waypoint_order)
    
    return total_spline_path_distance

def calculate_total_path_vertical_distance(positions, waypoint_order):
    '''
    Calculates the total vertical distance (sum of all altitude changes) of waypoints in a route.

    Parameters
    ----------
    positions : numpy.array
        An array of (x, y, z) coordinates for all the waypoints. 
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).
    waypoint_order : numpy.array
        An array of indices that specifies the order in which waypoints are visited.

    Returns
    ----------
    distance : int or float
        The total vertical distance (sum of all altitude changes) of a route.
    '''

    # Could have used numpy's cool fancy indexing, positions_in_route_order = positions[waypoint_order].
    # Still preferred to use regular accessing. The above creates a copy of positions which might
    # become expensive if positions is very large.
    waypoints_in_waypoint_order = [positions[i] for i in waypoint_order]
    total_vertical_distance = helper_calculate_total_vertical_distance(waypoints_in_waypoint_order)

    return total_vertical_distance

def calculate_total_spline_vertical_distance(spline_x_pts, spline_y_pts, spline_z_pts):
    '''
    Calculates the total vertical distance (sum of all altitude changes) of waypoints in a route.

    Parameters
    ----------
    spline_x_pts : numpy.array
        The x-coordinates for the spline curve.
    spline_y_pts : numpy.array
        The y-coordinates for the spline curve.
    spline_z_pts : numpy.array
        The z-coordinates for the spline curve.

    Returns
    ----------
    distance : int or float
        The total vertical distance (sum of all altitude changes) of a spline route.
    '''
    waypoints_in_waypoint_order = format_spline_pts(spline_x_pts, spline_y_pts, spline_z_pts)
    total_vertical_distance = helper_calculate_total_vertical_distance(waypoints_in_waypoint_order)

    return total_vertical_distance

def calculate_cumulative_path_rotation_angle(positions, waypoint_order, count_number_of_sharp_corners=False):
    '''
    Calculates the cumulative path rotation angle.

    Parameters
    ----------
    positions : numpy.array
        An array of (x, y, z) coordinates for all the waypoints. 
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).
    waypoint_order : numpy.array
        An array of indices that specifies the order in which waypoints are visited.
    (optional) number_of_sharp_corners : boolean
        A boolean value that determines if the number of sharp corners (angle > 90 degrees), in the flight path, should be counted.
        
    Returns
    ----------
    cumulative_rotation_angle : int or float
        The cumulative path rotation angle (in radians).
    (optional) number_of_sharp_corners : int
        The number of sharp corners (angle > 90 degrees) in the flight path.
    '''
    waypoints_in_waypoint_order = [positions[i] for i in waypoint_order]
    
    return helper_calculate_total_rotation_angle(waypoints_in_waypoint_order, count_number_of_sharp_corners)

def calculate_cumulative_spline_rotation_angle(spline_x_pts, spline_y_pts, spline_z_pts, count_number_of_sharp_corners=False):
    '''
    Calculates the cumulative path rotation angle.

    Parameters
    ----------
    spline_x_pts : numpy.array
        The x-coordinates for the spline curve.
    spline_y_pts : numpy.array
        The y-coordinates for the spline curve.
    spline_z_pts : numpy.array
        The z-coordinates for the spline curve.
    (optional) number_of_sharp_corners : int
        A boolean value that determines if the number of sharp corners (angle > 90 degrees), in the flight path, should be counted.
        
    Returns
    ----------
    cumulative_rotation_angle : int or float
        The cumulative path rotation angle in radians).
    (optional) number_of_sharp_corners : int
        The number of sharp corners (angle > 90 degrees) in the flight path.
    '''
    waypoints_in_waypoint_order = format_spline_pts(spline_x_pts, spline_y_pts, spline_z_pts)
    
    return helper_calculate_total_rotation_angle(waypoints_in_waypoint_order, count_number_of_sharp_corners)

def calculate_path_flight_duration_estimate(positions, 
                                       waypoint_order, 
                                       flight_time_to_first_waypoint_s=0, 
                                       cruising_speed_ms=5, 
                                       hover_time_at_waypoint=3, 
                                       flight_time_back_to_base=0):
    '''
    Calculates a flight duration estimate for a path (order of waypoints).
    For flight duration estimation, assume a constant UAV cruising speed 
    (e.g., 5 m/s) and include appropriate hover time (e.g., 2-3 seconds) at 
    each waypoint. Flight duration is assumed to start at the first waypoint.
    
    So true_flight_duration = flight_time_to_w0 + flight_duration + flight_time_to_base.

    Parameters
    ----------
    positions : numpy.array
        An array of (x, y, z) coordinates for all the waypoints. 
        It has the form np.array([x_1, y_1, z_1], [x_2, y_2, z_2], ..., [x_n, y_n, z_n]).
    waypoint_order : numpy.array
        An array of indices that specifies the order in which waypoints are visited.
    (optional) flight_time_to_first_waypoint_s
        The total time in seconds it would take for the drone to takeoff and fly to the first waypoint in the route.
        The default is 0s.
    (optional) cruising_speed_ms
        The cruising speed of the drone in m/s. 
        The default is 5m/s
    (optional) hover_time_at_waypoint
        The amount of time the drone hovers at a waypoint in seconds.
        The default is 3s.
    (optional) flight_time_back_to_base
        The total time in seconds it would take for the drone to fly from the last waypoint back to where the operator tells it to go (usually base).
        The default is 0s.
        
    Returns
    ----------
    time : int or float
        The estimated flight duration time in seconds.
    '''
    total_path_distance = calculate_total_path_distance(positions, waypoint_order)
    flight_time = helper_calculate_flight_duration(waypoint_order,
                                                   total_path_distance,
                                                   flight_time_to_first_waypoint_s,
                                                   cruising_speed_ms, hover_time_at_waypoint,
                                                   flight_time_back_to_base)

    return  flight_time

def calculate_spline_flight_duration_estimate(waypoint_order,
                                                spline_x_pts,
                                                spline_y_pts,
                                                spline_z_pts, 
                                                flight_time_to_first_waypoint_s=0, 
                                                cruising_speed_ms=5, 
                                                hover_time_at_waypoint=3, 
                                                flight_time_back_to_base=0):
    '''
    Calculates a flight duration estimate for a path (order of waypoints).
    For flight duration estimation, assume a constant UAV cruising speed 
    (e.g., 5 m/s) and include appropriate hover time (e.g., 2-3 seconds) at 
    each waypoint. Flight duration is assumed to start at the first waypoint.
    
    So true_flight_duration = flight_time_to_w0 + flight_duration + flight_time_to_base.

    Parameters
    ----------
    spline_x_pts : numpy.array
        The x-coordinates for the spline curve.
    spline_y_pts : numpy.array
        The y-coordinates for the spline curve.
    spline_z_pts : numpy.array
        The z-coordinates for the spline curve.
    waypoint_order : numpy.array
        An array of indices that specifies the order in which waypoints are visited.
    (optional) flight_time_to_first_waypoint_s
        The total time in seconds it would take for the drone to takeoff and fly to the first waypoint in the route.
        The default is 0s.
    (optional) cruising_speed_ms
        The cruising speed of the drone in m/s. 
        The default is 5m/s
    (optional) hover_time_at_waypoint
        The amount of time the drone hovers at a waypoint in seconds.
        The default is 3s.
    (optional) flight_time_back_to_base
        The total time in seconds it would take for the drone to fly from the last waypoint back to where the operator tells it to go (usually base).
        The default is 0s.

    Returns
    ----------
    time : int or float
        The estimated flight duration time in seconds.
    '''
    total_path_distance = calculate_total_spline_distance(spline_x_pts, spline_y_pts, spline_z_pts)
    flight_time = helper_calculate_flight_duration(waypoint_order, total_path_distance, 
                                                   flight_time_to_first_waypoint_s, 
                                                   cruising_speed_ms, 
                                                   hover_time_at_waypoint, 
                                                   flight_time_back_to_base)

    return  flight_time

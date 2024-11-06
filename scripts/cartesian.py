#!/usr/bin/env python3

#%%
import numpy as np
import rospy
from pydrake.all import *

from lfd_smoother.core.trajectory_optimizer import TrajectoryOptimizer
from lfd_smoother.util.robot import FR3Drake
from lfd_smoother.util.config import FR3Config

from geometry_msgs.msg import Pose, Point, Quaternion



class CartesianOptimizer(TrajectoryOptimizer):

    def __init__(self, robot, config):
        self.robot = robot
        self.config = config
    
    def run(self):
        pass

    def create_hexagon_waypoints(self, start_pose, side_length, samples_per_edge):
        self.waypoints = []
        
        start_pose.position.x += side_length  
        current_pose = copy.deepcopy(start_pose)

        angles = [i * math.pi / 3.0 for i in range(7)]  # 0, 60, 120, 180, 240, 300 degrees
        
        # Generate self.waypoints for each vertex of the hexagon
        for j, angle in enumerate(angles):
            vertex_pose = copy.deepcopy(current_pose)
            vertex_pose.position.x = start_pose.position.x - side_length  * math.cos(angle)
            vertex_pose.position.y = start_pose.position.y - side_length * math.sin(angle)
            vertex_pose.position.z = start_pose.position.z
            if j>0:
                x_edge = np.linspace(self.waypoints[-1].position.x, 
                                    vertex_pose.position.x, samples_per_edge)[1:-1]
                y_edge = np.linspace(self.waypoints[-1].position.y, 
                                    vertex_pose.position.y, samples_per_edge)[1:-1]
                for i in range(len(x_edge)):
                    sample_pose = copy.deepcopy(current_pose)
                    sample_pose.position.x = x_edge[i]
                    sample_pose.position.y = y_edge[i] 
                    self.add_waypoint(sample_pose)
            self.add_waypoint(vertex_pose)
            
    
    def add_waypoint(self, waypoint):
        self.waypoints.append(self.robot.create_waypoint("wp", np.array([waypoint.position.x, 
                                                                         waypoint.position.y, 
                                                                         waypoint.position.z])))   
        self.waypoints[-1].set_rotation(Quaternion(waypoint.orientation.x,
                                                    waypoint.orientation.y,
                                                    waypoint.orientation.z,
                                                    waypoint.orientation.w))

#%%

class CartesianPlanner:

    def __init__(self, config):
        self.config = config
        self.demo = None

    def run(self):
        if self.config["robot_type"] == "fr3":
            self.robot = FR3Drake(pkg_xml=self.config["pkg_xml"],
                            urdf_path=self.config["urdf_path"])
            config = FR3Config(self.robot, self.demo)

        self.optimizer = CartesianOptimizer(self.robot, config)
        self.example_run()
        # self.optimizer.run()
    
    def example_run(self):
        start_position = Point(0, 0, 0)
        start_orientation = Quaternion(1, 0, 0, 0)
        start_pose = Pose(start_position, start_orientation)
        side_length = 0.1
        samples_per_edge = 5
        self.optimizer.create_hexagon_waypoints(start_pose, side_length, samples_per_edge)        
        
    def export_raw(self):
        ts = np.linspace(0, self.result_traj.end_time(), 1000)
        
        position = []
        velocity = []
        acceleration = []
        
        for t in ts:
            position.append(self.result_traj.value(t))
            velocity.append(self.result_traj.MakeDerivative().value(t))
            acceleration.append(self.result_traj.MakeDerivative(2).value(t))

        return ts, np.array(position), np.array(velocity), np.array(acceleration)


#%%

if __name__ == '__main__':

    rospy.init_node('trajectory_smoother_node')
    config = rospy.get_param("~smoother_config")
    planner = CartesianPlanner(config)
    planner.run()




smoother_config= {
"demo_filter_threshold": 0.01,
"pkg_xml": "/home/abrk/catkin_ws/src/lfd/lfd_smoothing/drake/franka_drake/package.xml",
"urdf_path": "package://franka_drake/urdf/fr3_nohand.urdf",
"config_hierarchy": ["config/opt/initial.yaml", "config/opt/main.yaml"],
"robot_type": "fr3"
}

optimizer = CartesianPlanner(smoother_config)
optimizer.run()
optimizer.robot.visualize()
#%%
start_position = Point(0, 0, 0)
start_orientation = Quaternion(1, 0, 0, 0)
start_pose = Pose(start_position, start_orientation)
side_length = 0.1
samples_per_edge = 5

optimizer.optimizer.create_hexagon_waypoints(start_pose, side_length, samples_per_edge)


#%%


trajopt = smoother.smoother.trajopts[0]

traj = trajopt.ReconstructTrajectory(smoother.smoother.result)

smoother.smoother.plot_trajectory(traj)

#%% 

import pickle

serialized_trajectory = pickle.dumps(traj)


newtraj = pickle.loads(serialized_trajectory)

smoother.smoother.plot_trajectory(newtraj)

#%% normalize trajectory

duration = traj.end_time() - traj.start_time()

basis = traj.basis()
scaled_knots = [(knot / duration) for knot in basis.knots()]

ntraj = BsplineTrajectory(
    BsplineBasis(basis.order(), scaled_knots),
    traj.control_points())

smoother.smoother.plot_trajectory(ntraj)

#%% Store JSON




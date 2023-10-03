import numpy as np
import pickle
import json
import rospy
from lfd_interface.msg import PoseTrajectoryPoint
from trajectory_msgs.msg import JointTrajectoryPoint

from lfd_storage.smoother_storage import SmootherStorage

from pydrake.all import *

class Demonstration:

    def __init__(self):
        pass
    
    def filter(self, thr_translation = 0.1, thr_angle = 0.1):
        # Append the start point manually
        indices = [0]
        for i in range(1,self.length):  
            translation, angle = self.pose_diff(self.positions[indices[-1]], self.orientations[indices[-1]], 
                                                self.positions[i], self.orientations[i])
            if translation > thr_translation or angle > thr_angle:
                indices.append(i)
        
        # Append the goal point manually
        indices.append(self.length - 1)

        self.apply_custom_index(indices)
    
    def apply_custom_index(self, indices):
        self.ys = self.ys[indices]
        self.ts = self.ts[indices]
        self.positions = self.positions[indices]
        self.orientations = self.orientations[indices]
        self.length = len(indices)

    def pose_diff(self, p1, q1, p2, q2):
        rb1 = RigidTransform(p1)
        rb1.set_rotation(Quaternion(q1))

        rb2 = RigidTransform(p2)
        rb2.set_rotation(Quaternion(q2))
        
        angle = np.abs(rb1.rotation().ToAngleAxis().angle() - rb2.rotation().ToAngleAxis().angle())
        translation = np.linalg.norm(rb1.translation() - rb2.translation())
        return translation, angle
    
    def divisible_by(self, n, overlap=0):
        indices = [i for i in range(self.length)]
        total_elements = len(indices)
        segments_count = (total_elements - overlap) // (n - overlap)
        elements_to_remove = total_elements - (segments_count * (n - overlap) + overlap)
        if elements_to_remove != 0:
            for _ in range(elements_to_remove):
                index_to_remove = len(indices) // 2  # Find the index of the middle element
                indices.pop(index_to_remove)  # Remove the value at the specified index

            self.apply_custom_index(indices)
        
    def reshape(self, step, overlap=0):
        self.ys = self.split_into_segments(self.ys, step, overlap)
        self.positions = self.split_into_segments(self.positions, step, overlap)
        self.orientations = self.split_into_segments(self.orientations, step, overlap)
        self.num_segments = self.ys.shape[0]

    def split_into_segments(self, waypoints, n, overlap=0):
        segments = []
        start = 0
        while start + n <= len(waypoints):
            segment = waypoints[start:start+n]
            segments.append(segment)
            start += n - overlap
        return np.array(segments)
    
    def read_from_pickle(self,filename):
        with open(filename, 'rb') as file:
            demonstration = pickle.load(file)
        
        self.read_from_ros(demonstration)

    def read_from_ros(self,demonstration):    
        joint_trajectory = demonstration.joint_trajectory

        n_time_steps = len(joint_trajectory.points)
        n_dim = len(joint_trajectory.joint_names)

        ys = np.zeros([n_time_steps,n_dim])
        ts = np.zeros(n_time_steps)

        for (i,point) in enumerate(joint_trajectory.points):
            ys[i,:] = point.positions
            ts[i] = point.time_from_start.to_sec()


        pose_trajectory = demonstration.pose_trajectory

        n_time_steps = len(joint_trajectory.points)

        positions = np.zeros((n_time_steps, 3))
        orientations = np.zeros((n_time_steps, 4))

        for (i,point) in enumerate(pose_trajectory.points):
            positions[i,:] = [point.pose.position.x, point.pose.position.y, point.pose.position.z]
            orientations[i,:] = [point.pose.orientation.w, point.pose.orientation.x, point.pose.orientation.y, point.pose.orientation.z] 
            
        # ys = np.insert(ys,[ys.shape[1], ys.shape[1]], 0, axis=1) 
        
        self.read_raw(ts,ys,positions,orientations)
    
    
    
    def export_waypoints(self, demo_name):
        storage = SmootherStorage(demo_name)
        obj = {"ts" : self.ts.tolist(),
               "ys" : self.ys.tolist(),
               "positions" : self.positions.tolist(),
               "orientations" : self.orientations.tolist()
               }
        storage.export_waypoints(obj)
     
            
    def read_from_waypoints(self, demo_name):
        storage = SmootherStorage(demo_name)

        obj = storage.import_waypoints()
        
        self.read_raw(np.array(obj["ts"]),
                      np.array(obj["ys"]),
                      np.array(obj["positions"]),
                      np.array(obj["orientations"]))
    
    def read_raw(self, ts, ys, positions, orientations):
        self.ts = ts
        self.ys = ys
        self.positions = positions
        self.orientations = orientations
        (self.length, self.num_q) = ys.shape 


    def export_to_ros(self, demo_template):
        # Prepare JointTrajectory
        joint_trajectory = demo_template.joint_trajectory
        joint_trajectory.points = []
        for i in range(self.length):
            point = JointTrajectoryPoint()  # Get the type of the points and instantiate
            point.positions = self.ys[i, :]
            point.time_from_start = rospy.Duration.from_sec(self.ts[i]) 
            joint_trajectory.points.append(point)

        # Prepare PoseTrajectory
        pose_trajectory = demo_template.pose_trajectory
        pose_trajectory.points = []
        for i in range(self.length):
            point = PoseTrajectoryPoint()  # Get the type of the points and instantiate
            point.pose.position.x = self.positions[i, 0]
            point.pose.position.y = self.positions[i, 1]
            point.pose.position.z = self.positions[i, 2]
            point.pose.orientation.w = self.orientations[i, 0]
            point.pose.orientation.x = self.orientations[i, 1]
            point.pose.orientation.y = self.orientations[i, 2]
            point.pose.orientation.z = self.orientations[i, 3]
            pose_trajectory.points.append(point)

        # Put together in the DemonstrationMsg
        demo_template.joint_trajectory = joint_trajectory
        demo_template.pose_trajectory = pose_trajectory

        return demo_template

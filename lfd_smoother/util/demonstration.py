import numpy as np
import pickle
import json

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
        
        self.ts = ts
        self.ys = ys
        self.positions = positions
        self.orientations = orientations
        (self.length, self.num_q) = ys.shape
    
    
    def export_as_json(self, filename):
        obj = {"ts" : self.ts.tolist(),
               "ys" : self.ys.tolist(),
               "positions" : self.positions.tolist(),
               "orientations" : self.orientations.tolist()
               }
        with open(filename, 'w') as file:
            file.write(json.dumps(obj))
     
            
    def read_from_json(self, filename):
        with open(filename, "r") as file:
            obj = json.loads(file.read())
        
        self.ts = np.array(obj["ts"])
        self.ys = np.array(obj["ys"])
        self.positions = np.array(obj["positions"])
        self.orientations = np.array(obj["orientations"])
        (self.length, self.num_q) = self.ys.shape
        
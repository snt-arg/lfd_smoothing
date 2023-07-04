#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:14:08 2023

@author: abrk
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from pydrake.all import *

from manipulation import running_as_notebook
from manipulation.meshcat_utils import PublishPositionTrajectory
from manipulation.scenarios import AddIiwa, AddPlanarIiwa, AddShape, AddWsg
from manipulation.utils import ConfigureParser

#%% Drake Util Class

class DrakeUtil:

    def __init__(self):
        self.meshcat = StartMeshcat()
        self.meshcat.Delete()
        self.builder = DiagramBuilder()

        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.001)
        self.fr3 = self.add_fr3()

        parser = Parser(self.plant)
        self.plant.Finalize()

        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            self.meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration),
        )
        self.collision_visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            self.meshcat,
            MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
        )
        self.meshcat.SetProperty("collision", "visible", False)

        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        self.num_q = self.plant.num_positions()
        self.q0 = self.plant.GetPositions(self.plant_context)
        self.gripper_frame = self.plant.GetFrameByName("fr3_link8", self.fr3)
 
    
    def add_fr3(self):
        urdf = "package://my_drake/manipulation/models/franka_description/urdf/fr3.urdf"

        parser = Parser(self.plant)
        parser.package_map().AddPackageXml("/home/abrk/catkin_ws/src/franka/franka_ros/franka_description/package.xml")
        parser.package_map().AddPackageXml("/home/abrk/thesis/drake/my_drake/package.xml")
        fr3 = parser.AddModelsFromUrl(urdf)[0]
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName("fr3_link0"))

        # Set default positions:
        q0 = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78, 0.035, 0.035]
        index = 0
        for joint_index in self.plant.GetJointIndices(fr3):
            joint = self.plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q0[index])
                index += 1

        return fr3
    
    def create_waypoint(self, name, position):
        X = RigidTransform(position)
        self.meshcat.SetObject(name, Sphere(0.003), rgba=Rgba(0.9, 0.1, 0.1, 1))
        self.meshcat.SetTransform(name, X)
        return X

#%% Trajectory Smoother Class

class TrajectoryOptimizer:

    def __init__(self, drake_util,ys, ts, positions, orientations, num_control_points = 4, bspline_order = 4, thr_translation = 0.1, thr_angle = 0.1):
        self.drake_util = drake_util
        self.num_control_points = num_control_points
        self.bspline_order = bspline_order
        self.waypoints = []
        filtered_indices = self.filter_input(positions, orientations, thr_translation, thr_angle)
        self.ys = ys[filtered_indices]
        self.ts = ts[filtered_indices]
        self.positions = positions[filtered_indices]
        self.orientations = orientations[filtered_indices]
        self.set_waypoints(self.positions, self.orientations)
    
    def run_with_joint_constraints(self):
        self.init_trajopts()
        self.create_sym_r()
        self.apply_joint_constraints()
        self.apply_joining_constraints()
        self.solve()
        self.plot_trajectory(self.append_trajectories())

    def run_with_task_constraints(self):
        self.init_trajopts()
        self.create_sym_r()
        self.apply_task_constraints()
        self.apply_joining_constraints()
        self.solve()
        self.plot_trajectory(self.append_trajectories())

    def filter_input(self, positions, orientations, thr_translation = 0.1, thr_angle = 0.1):
        indices = [0]
        for i in range(1,positions.shape[0]):  
            translation, angle = self.pose_diff(positions[indices[-1]], orientations[indices[-1]], positions[i], orientations[i])
            if translation > thr_translation or angle > thr_angle:
                indices.append(i)  
        return indices

    def create_sym_r(self):
        self.sym_r = []
        self.sym_rvel = []
        self.sym_racc = []
        self.sym_rjerk = []
        for trajopt in self.trajopts:
            control_points = trajopt.control_points()
            ufunc = np.vectorize(Expression)
            control_points_exp = [ufunc(control_points[:,i,np.newaxis]) for i in range(self.num_control_points)]
            basis_exp = BsplineBasis_[Expression](trajopt.basis().order(), trajopt.basis().knots())  
            sym_r = BsplineTrajectory_[Expression](basis_exp,control_points_exp)
            
            self.sym_r.append(sym_r)
            self.sym_rvel.append(sym_r.MakeDerivative())
            self.sym_racc.append(sym_r.MakeDerivative(2))
            self.sym_rjerk.append(sym_r.MakeDerivative(3))
    
    def init_trajopts(self, duration_coeff = 1.0, path_length_coeff = 0.0, duration_bound = [0.01, 5]):
        self.trajopts = []
        self.progs = []
        plant = self.drake_util.plant

        for i in range(len(self.waypoints)-1):
            self.trajopts.append(KinematicTrajectoryOptimization(plant.num_positions(), 
                                                                self.num_control_points, spline_order=self.bspline_order))

        for trajopt in self.trajopts:
            self.progs.append(trajopt.get_mutable_prog())
            trajopt.AddDurationCost(duration_coeff)
            # trajopt.AddPathLengthCost(path_length_coeff)
            trajopt.AddPositionBounds(
                plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
            )
            trajopt.AddVelocityBounds(
                plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
            )
            trajopt.AddDurationConstraint(duration_bound[0], duration_bound[1])
    
    def apply_quadratic_velocity_cost(self, coeff=1.0):
        
        for i in range(len(self.trajopts)):
            cps = self.sym_rvel[i].control_points()
            for j in range(len(cps)):
                self.progs[i].AddCost(matmul(cps[j].transpose(),cps[j])[0,0] * coeff / pow(self.trajopts[i].duration(),2))

    def apply_quadratic_jerk_cost(self, coeff=1.0):
        
        for i in range(len(self.trajopts)):
            # j0 = self.sym_rjerk[i].value(0.5)
            # # j1 = self.sym_rjerk[i].value(1)
            # self.progs[i].AddCost(matmul(j0.transpose(), j0)[0,0] * coeff / pow(self.trajopts[i].duration(),6))
           
            cps = self.sym_rjerk[i].control_points()
            for j in range(len(cps)):
                self.progs[i].AddCost(matmul(cps[j].transpose(),cps[j])[0,0] * coeff / pow(self.trajopts[i].duration(),6))
    
    def apply_quadratic_acceleration_cost(self, coeff=1.0):
        
        for i in range(len(self.trajopts)):
            cps = self.sym_racc[i].control_points()
            for j in range(len(cps)):
                self.progs[i].AddCost(matmul(cps[j].transpose(),cps[j])[0,0] * coeff / pow(self.trajopts[i].duration(),4))

    def apply_joint_constraints(self, lb = 0, ub = 0):
        num_q = self.drake_util.plant.num_positions()

        self.trajopts[0].AddPathPositionConstraint(self.ys[0], self.ys[0] ,0)
        self.trajopts[-1].AddPathPositionConstraint(self.ys[-1],self.ys[-1],1)

        self.trajopts[0].AddPathVelocityConstraint(
            np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
        )

        self.trajopts[-1].AddPathVelocityConstraint(
            np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
        )

        for i in range(1,len(self.ys)-1):
            self.trajopts[i-1].AddPathPositionConstraint(self.ys[i] - lb, self.ys[i] + ub, 1)
            self.trajopts[i].AddPathPositionConstraint(self.ys[i] - lb, self.ys[i] + ub, 0)
    
    def apply_joint_cost(self, coeff):
        num_q = self.drake_util.plant.num_positions()
        for i in range(1,len(self.ys)-1):
            self.progs[i].AddQuadraticErrorCost(
                coeff*np.eye(num_q), self.ys[i], self.trajopts[i].control_points()[:, 0]
            )
    
    def apply_task_constraints(self, tol_translation=0.01, tol_rotation=0.1):
        plant = self.drake_util.plant
        gripper_frame = self.drake_util.gripper_frame
        plant_context = self.drake_util.plant_context
        num_q = self.drake_util.plant.num_positions()

        start_constraint = PositionConstraint(
            plant,
            plant.world_frame(),
            self.waypoints[0].translation(),
            self.waypoints[0].translation(),
            gripper_frame,
            [0, 0, 0],
            plant_context,
        )
        self.trajopts[0].AddPathPositionConstraint(start_constraint, 0)

        ori_const = OrientationConstraint(
            plant, 
            gripper_frame, 
            RotationMatrix(),
            plant.world_frame(),
            self.waypoints[0].rotation(),
            0.0,
            plant_context
            )

        self.trajopts[0].AddPathPositionConstraint(ori_const, 0)

        self.trajopts[0].AddPathVelocityConstraint(
            np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
        )


        # progs[0].AddQuadraticErrorCost(
        #     1*np.eye(num_q), ys[0], trajopts[0].control_points()[:, 0]
        # )


        # Goal Constraints
        goal_constraint = PositionConstraint(
            plant,
            plant.world_frame(),
            self.waypoints[-1].translation(),
            self.waypoints[-1].translation(),
            gripper_frame,
            [0, 0, 0],
            plant_context,
        )
        self.trajopts[-1].AddPathPositionConstraint(goal_constraint, 1)

        ori_const = OrientationConstraint(
            plant, 
            gripper_frame, 
            RotationMatrix(),
            plant.world_frame(),
            self.waypoints[-1].rotation(),
            0.0,
            plant_context
            )

        self.trajopts[-1].AddPathPositionConstraint(ori_const, 1)

        self.trajopts[-1].AddPathVelocityConstraint(
            np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
        )

        # progs[-1].AddQuadraticErrorCost(
        #     1*np.eye(num_q), ys[-1], trajopts[-1].control_points()[:, -1]
        # )


        # Middle Constraints

        for i in range(1,len(self.waypoints)-1):
            middle_constraint = PositionConstraint(
                plant,
                plant.world_frame(),
                self.waypoints[i].translation() - tol_translation,
                self.waypoints[i].translation() + tol_translation,
                gripper_frame,
                [0, 0, 0],
                plant_context,
            )
            self.trajopts[i-1].AddPathPositionConstraint(middle_constraint, 1)
            # self.trajopts[i].AddPathPositionConstraint(middle_constraint, 0)
            
            ori_const = OrientationConstraint(
                plant, 
                gripper_frame, 
                RotationMatrix(),
                plant.world_frame(),
                self.waypoints[i].rotation(),
                tol_rotation,
                plant_context
                )
            
            self.trajopts[i-1].AddPathPositionConstraint(ori_const, 1)
            # self.trajopts[i].AddPathPositionConstraint(ori_const, 0)
            
            # progs[i-1].AddQuadraticErrorCost(
            #     1*np.eye(num_q), ys[i], trajopts[i-1].control_points()[:, -1]
            # )
            # progs[i].AddQuadraticErrorCost(
            #     1*np.eye(num_q), ys[i], trajopts[i].control_points()[:, 0]
            # )

    
    def apply_joining_constraints(self):
        num_q = self.drake_util.plant.num_positions()

        self.nprog = MathematicalProgram()

        for prog in self.progs:
            self.nprog.AddDecisionVariables(prog.decision_variables())
            for const in prog.GetAllConstraints():
                self.nprog.AddConstraint(const.evaluator(), const.variables())
            for cost in prog.GetAllCosts():
                self.nprog.AddCost(cost.evaluator(), cost.variables())
            

        for i in range(1,len(self.ys)-1):
            for j in range(num_q):
                self.nprog.AddConstraint(((self.sym_rvel[i-1].value(1) / self.trajopts[i-1].duration()) 
                                    - (self.sym_rvel[i].value(0) / self.trajopts[i].duration()))[j][0]
                                    , 0, 0)
                self.nprog.AddConstraint(((self.sym_racc[i-1].value(1) / self.trajopts[i-1].duration() / self.trajopts[i-1].duration()) 
                                    - (self.sym_racc[i].value(0) / self.trajopts[i].duration() / self.trajopts[i].duration()))[j][0]
                                    , 0, 0)
                # self.nprog.AddConstraint(((self.sym_rjerk[i-1].value(1) / self.trajopts[i-1].duration() / self.trajopts[i-1].duration() / self.trajopts[i-1].duration()) 
                #                     - (self.sym_rjerk[i].value(0) / self.trajopts[i].duration() / self.trajopts[i].duration() / self.trajopts[i].duration()))[j][0]
                #                     , 0, 0)
            self.nprog.AddLinearConstraint(self.sym_r[i-1].value(1) - self.sym_r[i].value(0), np.zeros((num_q, 1)), np.zeros((num_q, 1)))

    def solve(self):
        self.result = Solve(self.nprog)
        self.success = self.result.is_success()

        if not self.success:
            print("optimization failed")
        else:
            print("optimizer finished successfully")
        
    def append_trajectories(self):
        output_trajs = []
        t_offset = 0.0
        for trajopt in self.trajopts:
            duration, traj = self.reconstruct_trajectory(self.result, trajopt, t_offset)
            output_trajs.append(traj)
            t_offset += duration
            
        self.output_trajs = output_trajs
        return CompositeTrajectory(output_trajs)


    def set_waypoints(self, positions, orientations):
        for i in range(0,positions.shape[0]):  
            self.waypoints.append(self.drake_util.create_waypoint("wp{}".format(i), positions[i]))   
            self.waypoints[-1].set_rotation(Quaternion(orientations[i]))
    
    def reconstruct_trajectory(self, result, trajopt, t_offset):
        duration = result.GetSolution(trajopt.duration())
        basis = trajopt.basis()
        scaled_knots = [(knot * duration) + t_offset for knot in basis.knots()]
        
        return duration, BsplineTrajectory(
            BsplineBasis(basis.order(), scaled_knots),
            [result.GetSolution(trajopt.control_points())[:,i,np.newaxis] for i in range(self.num_control_points)])
    
    def plot_trajectory(self, composite_traj):

        PublishPositionTrajectory(
            composite_traj, self.drake_util.context, self.drake_util.plant, self.drake_util.visualizer
        )
        self.drake_util.collision_visualizer.ForcedPublish(
            self.drake_util.collision_visualizer.GetMyContextFromRoot(self.drake_util.context)
        )

        ts = np.linspace(0, composite_traj.end_time(), 1000)
        
        position = []
        velocity = []
        acceleration = []
        jerk = []
        
        for t in ts:
            position.append(composite_traj.value(t))
            velocity.append(composite_traj.MakeDerivative().value(t))
            acceleration.append(composite_traj.MakeDerivative(2).value(t))
            jerk.append(composite_traj.MakeDerivative(3).value(t))
        
        # Plot position, velocity, and acceleration
        fig, axs = plt.subplots(1, 4, figsize=(50,10))
        
        # Position plot
        axs[0].plot(ts, np.array(position).squeeze(axis=2))
        axs[0].set_ylabel('Position')
        axs[0].set_xlabel('Time')
        
        # Velocity plot
        axs[1].plot(ts, np.array(velocity).squeeze(axis=2))
        axs[1].set_ylabel('Velocity')
        axs[1].set_xlabel('Time')
        
        # Acceleration plot
        axs[2].plot(ts, np.array(acceleration).squeeze(axis=2))
        axs[2].set_ylabel('Acceleration')
        axs[2].set_xlabel('Time')
        
        # Jerk plot
        axs[3].plot(ts, np.array(jerk).squeeze(axis=2))
        axs[3].set_ylabel('Jerk')
        axs[3].set_xlabel('Time')
        
        plt.tight_layout()
        plt.show()    

    def pose_diff(self, p1, q1, p2, q2):
        rb1 = RigidTransform(p1)
        rb1.set_rotation(Quaternion(q1))

        rb2 = RigidTransform(p2)
        rb2.set_rotation(Quaternion(q2))
        
        angle = np.abs(rb1.rotation().ToAngleAxis().angle() - rb2.rotation().ToAngleAxis().angle())
        translation = np.linalg.norm(rb1.translation() - rb2.translation())
        return translation, angle
    
    def export_initial_guess(self):
        initial_guess = []
        for trajopt in self.trajopts:
            initial_guess.append(self.result.GetSolution(trajopt.control_points()))
        return initial_guess
    
    def apply_initial_guess(self, initial_guess):
        for (i,trajopt) in enumerate(self.trajopts):
            self.nprog.SetInitialGuess(trajopt.control_points(), initial_guess[i])
    
#%%
    
def read_data():
    with open('/home/abrk/catkin_ws/src/lfd/lfd_dmp/scripts/experimental/smoothing/0.pickle', 'rb') as file:
        demonstration = pickle.load(file)
    
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

    return ys, ts, positions, orientations   


#%%

thr_translation = 0.01

ys, ts, positions, orientations = read_data()
drake_util = DrakeUtil()

smoother = TrajectoryOptimizer(drake_util, ys, ts, positions, orientations,4,4, thr_translation=thr_translation)
smoother.run_with_joint_constraints()
initial_guess = smoother.export_initial_guess()

#%%

# smoother = TrajectoryOptimizer(drake_util, ys, ts, positions, orientations,6,4, thr_translation=thr_translation)
# smoother.init_trajopts(duration_coeff=10, path_length_coeff=0.0)
# smoother.create_sym_r()
# # smoother.apply_quadratic_velocity_cost(coeff=-5.0)
# smoother.apply_quadratic_jerk_cost(coeff=0.001/(len(smoother.trajopts) * 4))
# # smoother.apply_joint_cost(coeff=0.1)
# # smoother.apply_quadratic_acceleration_cost(coeff=10.0)

# # lb = np.array([-1,-1,-1,-1,-1,-1,-1,0,0])
# # ub = np.array([1,1,1,1,1,1,1,0,0])
# # smoother.apply_joint_constraints(lb=0.1*ub,ub=0.1*ub)
# smoother.apply_task_constraints(tol_translation=0.01, tol_rotation=0.2)
# smoother.apply_joining_constraints()
# smoother.apply_initial_guess(initial_guess)
# smoother.solve()
# smoother.plot_trajectory(smoother.append_trajectories())


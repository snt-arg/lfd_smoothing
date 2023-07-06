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
import copy

from pydrake.all import *

from manipulation.meshcat_utils import PublishPositionTrajectory


#%% Drake Util Class

class FR3Drake:

    def __init__(self, franka_pkg_xml, urdf_path, base_frame = "fr3_link0", gripper_frame = "fr3_link8", home_pos = None):
        
        self.urdf_path = urdf_path
        self.franka_path = franka_pkg_xml
        
        self.meshcat = StartMeshcat()
        self.meshcat.Delete()
        self.builder = DiagramBuilder()

        if home_pos is None:
            home_pos = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78, 0.035, 0.035]

        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.001)
        self.robot = self.add_fr3(home_pos, base_frame)

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
        self.gripper_frame = self.plant.GetFrameByName(gripper_frame, self.robot)
        
 
    
    def add_fr3(self, home_pos, base_frame):
        urdf = "package://franka_drake/urdf/fr3.urdf"

        parser = Parser(self.plant)
        # parser.package_map().AddPackageXml("/home/abrk/catkin_ws/src/franka/franka_ros/franka_description/package.xml")
        parser.package_map().AddPackageXml(self.franka_path)
        fr3 = parser.AddModelsFromUrl(self.urdf_path)[0]
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName(base_frame))

        # Set default positions:
        q0 = home_pos
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

class Demonstration:

    def __init__(self, ts, ys, positions, orientations):
        self.ts = ts
        self.ys = ys
        self.positions = positions
        self.orientations = orientations
        (self.length, self.num_q) = ys.shape
    
    def filter(self, thr_translation = 0.1, thr_angle = 0.1):
        # Append the start point manually
        indices = [0]
        for i in range(1,self.length):  
            translation, angle = self.pose_diff(positions[indices[-1]], orientations[indices[-1]], 
                                                positions[i], orientations[i])
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


class TrajectoryOptimizer:

    def __init__(self, robot, config : dict):
        self.robot = robot
        self.config = config
        self.num_control_points = config["num_cps"]
        self.bspline_order = config["bspline_order"]
        self.wp_per_segment = config.get("wp_per_segment", self.num_control_points)
        self.overlap = config.get("overlap", 0)
        self.duration_bound = config.get("bound_duration", [0.01,5])
        self.coeff_duration = config.get("coeff_duration", 1)
        self.coeff_jerk = config.get("coeff_jerk", 0.004)
        self.coeff_joint_cp_error = config.get("coeff_joint_cp_error", 1)
        self.tol_joint = config.get("tol_joint", 0)
        self.tol_translation = config.get("tol_translation", 0.02)
        self.tol_rotation = config.get("tol_rotation", 0.1)
        self.solver = config.get("solver", None)
        self.solver_log = config.get("solver_log", "/tmp/trajopt.txt")
        

    def input_demo(self, demonstration):

        self.demo = copy.deepcopy(demonstration)
        self.demo.divisible_by(self.wp_per_segment, self.overlap)
        self.waypoints = []
        self.set_waypoints(self.demo)
        self._segment_demo()
    
    def _segment_demo(self):
        step = self.wp_per_segment
        self.demo.reshape(step, self.overlap)
        self.waypoints = demo.split_into_segments(np.array(self.waypoints), 
                                                  step, self.overlap)
        self.num_segments = self.demo.num_segments

    def _make_symbolic(self):
        self.sym_r = []
        self.sym_rvel = []
        self.sym_racc = []
        self.sym_rjerk = []

        for trajopt in self.trajopts:
            sym_r = self._create_sym_r(trajopt)
            self.sym_r.append(sym_r)
            self.sym_rvel.append(sym_r.MakeDerivative())
            self.sym_racc.append(sym_r.MakeDerivative(2))
            self.sym_rjerk.append(sym_r.MakeDerivative(3))
    
    def _create_sym_r(self, trajopt):
        control_points = trajopt.control_points()
        ufunc = np.vectorize(Expression)
        control_points_exp = [ufunc(control_points[:,i,np.newaxis]) 
                              for i in range(self.num_control_points)]
        basis_exp = BsplineBasis_[Expression](trajopt.basis().order(), 
                                              trajopt.basis().knots())  
        return BsplineTrajectory_[Expression](basis_exp,control_points_exp)
    
    def add_pos_vel_bounds(self):
        for trajopt in self.trajopts:
            trajopt.AddPositionBounds(
                self.robot.plant.GetPositionLowerLimits(), 
                self.robot.plant.GetPositionUpperLimits()
            )
            trajopt.AddVelocityBounds(
                self.robot.plant.GetVelocityLowerLimits(),
                self.robot.plant.GetVelocityUpperLimits()
            )

    def add_duration_bound(self):
        for trajopt in self.trajopts:
            trajopt.AddDurationConstraint(self.duration_bound[0],
                                          self.duration_bound[1])


    def add_duration_cost(self):
        for trajopt in self.trajopts:
            trajopt.AddDurationCost(self.coeff_duration)


    def init_trajopts(self):
        self.trajopts = []
        self.progs = []
             
        for _ in range(self.num_segments):
            trajopt = KinematicTrajectoryOptimization(self.robot.plant.num_positions(), 
                                                    self.num_control_points,
                                                    spline_order=self.bspline_order)
            self.trajopts.append(trajopt)
            self.progs.append(trajopt.get_mutable_prog())
        
        self._make_symbolic()

    def add_jerk_cost(self):
        # This function would only be correct if num_cps==bspline_order==4
        for i in range(self.num_segments):
            cps = self.sym_rjerk[i].control_points()
            for j in range(len(cps)):
                self.progs[i].AddCost(matmul(cps[j].transpose(),cps[j])[0,0] * self.coeff_jerk 
                                      / pow(self.trajopts[i].duration(),5) # Order is one less than expected (6) because of Jdt
                                      / (len(cps)*self.num_segments*self.robot.plant.num_positions()))
    
    
    def _add_joint_constraint(self, trajopt, position, tolerance, ntime, rest=False):
        trajopt.AddPathPositionConstraint(position - tolerance, position + tolerance, ntime)
        if rest:
            self._add_zerovel_constraint(trajopt,ntime)

    def add_joint_constraints(self, tolerance = 0):

        self._add_joint_constraint(self.trajopts[0], self.demo.ys[0,0], 0, 0, rest=True)
        self._add_joint_constraint(self.trajopts[-1], self.demo.ys[-1,-1], 0, 1, rest=True)

        for i in range(0,self.num_segments - 1):
            self._add_joint_constraint(self.trajopts[i], self.demo.ys[i,-1], tolerance, 1)
    
    def add_joint_cp_error_cost(self, coeff):
        num_q = self.robot.plant.num_positions()
        if self.wp_per_segment == self.num_control_points:
            for i in range(0,self.num_segments):
                for j in range(1, self.num_control_points):
                    self.progs[i].AddQuadraticErrorCost(
                        coeff*np.eye(num_q), self.demo.ys[i,j], self.trajopts[i].control_points()[:, j]
                    )
        else:
            print ("num_control_points is not equal to wp_per_segment, ignoring joint_cp_error_cost")    

    def _add_zerovel_constraint(self,trajopt, ntime):
        num_q = self.robot.plant.num_positions()
        trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), ntime)

    def _add_pose_constraint(self, trajopt, waypoint, tol_translation, tol_rotation, ntime, rest=False):
        plant = self.robot.plant
        gripper_frame = self.robot.gripper_frame
        plant_context = self.robot.plant_context

        position_constraint = PositionConstraint(
            plant,
            plant.world_frame(),
            waypoint.translation() - tol_translation,
            waypoint.translation() + tol_translation,
            gripper_frame,
            [0, 0, 0],
            plant_context,
        )
        orientation_constraint = OrientationConstraint(
            plant, 
            gripper_frame, 
            RotationMatrix(),
            plant.world_frame(),
            waypoint.rotation(),
            tol_rotation,
            plant_context
            )
        
        trajopt.AddPathPositionConstraint(position_constraint, ntime)
        trajopt.AddPathPositionConstraint(orientation_constraint, ntime)

        if rest:
            self._add_zerovel_constraint(trajopt,ntime)       
 
    
    def add_task_constraints(self):

        self._add_joint_constraint(self.trajopts[0], self.demo.ys[0,0],0, 0, rest=True)
        self._add_joint_constraint(self.trajopts[-1], self.demo.ys[-1,-1],0, 1, rest=True)
        
        # self._add_pose_constraint(self.trajopts[0], self.waypoints[0,0],0, 0, 0, rest=True)
        # self._add_pose_constraint(self.trajopts[-1], self.waypoints[-1,-1],0, 0, 1, rest=True)

        for i in range(0,self.num_segments-1):
            self._add_pose_constraint(self.trajopts[i], self.waypoints[i,-1], 
                                      self.tol_translation, self.tol_rotation, 1)
    
    
    def finalize(self):
        num_q = self.robot.plant.num_positions()
        self.nprog = MathematicalProgram()

        for prog in self.progs:
            self.nprog.AddDecisionVariables(prog.decision_variables())
            for const in prog.GetAllConstraints():
                self.nprog.AddConstraint(const.evaluator(), const.variables())
            for cost in prog.GetAllCosts():
                self.nprog.AddCost(cost.evaluator(), cost.variables())
            
        for i in range(1,self.num_segments):
            for j in range(num_q):
                self.nprog.AddConstraint(((self.sym_rvel[i-1].value(1) / self.trajopts[i-1].duration()) 
                                    - (self.sym_rvel[i].value(0) / self.trajopts[i].duration()))[j][0]
                                    , 0, 0)
                self.nprog.AddConstraint(((self.sym_racc[i-1].value(1) / pow(self.trajopts[i-1].duration(),2)) 
                                    - (self.sym_racc[i].value(0) / pow(self.trajopts[i].duration(),2)))[j][0]
                                    , 0, 0)
            self.nprog.AddLinearConstraint(self.sym_r[i-1].value(1) - self.sym_r[i].value(0), np.zeros((num_q, 1)), np.zeros((num_q, 1)))

    def solve(self):
        self._solve(self.nprog)
    
    def _solve(self,prog):
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintFileName, self.solver_log)

        if self.solver is None:
            self.result = Solve(prog, solver_options=solver_options)
        else:
            self.result = self.solver.Solve(prog, solver_options=solver_options)
        self.success = self.result.is_success()
        if not self.success:
            print("Optimization Failed")
        else:
            print("Optimization Finished Successfully")
        
    def compile_trajectory(self):
        output_trajs = []
        t_offset = 0.0
        for trajopt in self.trajopts:
            duration, traj = self._reconstruct_trajectory(self.result, trajopt, t_offset)
            output_trajs.append(traj)
            t_offset += duration
            
        self.output_trajs = output_trajs
        return CompositeTrajectory(output_trajs)


    def set_waypoints(self, demo):
        for i in range(0, demo.length):  
            self.waypoints.append(self.robot.create_waypoint("wp{}".format(i), demo.positions[i]))   
            self.waypoints[-1].set_rotation(Quaternion(demo.orientations[i]))
    
    def _reconstruct_trajectory(self, result, trajopt, t_offset):
        duration = result.GetSolution(trajopt.duration())
        basis = trajopt.basis()
        scaled_knots = [(knot * duration) + t_offset for knot in basis.knots()]
        
        return duration, BsplineTrajectory(
            BsplineBasis(basis.order(), scaled_knots),
            [result.GetSolution(trajopt.control_points())[:,i,np.newaxis] for i in range(self.num_control_points)])
    
    def plot_trajectory(self, composite_traj):

        PublishPositionTrajectory(
            composite_traj, self.robot.context, self.robot.plant, self.robot.visualizer
        )
        self.robot.collision_visualizer.ForcedPublish(
            self.robot.collision_visualizer.GetMyContextFromRoot(self.robot.context)
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
    
    def export_initial_guess(self):
        initial_guess = []
        for trajopt in self.trajopts:
            initial_guess.append(self.result.GetSolution(trajopt.control_points()))
        return initial_guess
    
    def apply_initial_guess(self, initial_guess):
        for (i,trajopt) in enumerate(self.trajopts):
            self.nprog.SetInitialGuess(trajopt.control_points(), initial_guess[i])

    def export_timing(self):
        timings = [0]
        for trajopt in self.trajopts:
            timings.append(timings[-1] + self.result.GetSolution(trajopt.duration()))
        return timings
    
    def import_timing(self, timings):
        self.timings = timings
#%%
class SingleOptimizer(TrajectoryOptimizer):

    def __init__(self, robot, config):
        super().__init__(robot, config)
        self.coeff_vel = config.get("coeff_vel", 1)

        
    def apply_initial_guess(self, initial_guess):
        trajopt = self.trajopts[0]
        initial_guess = np.array(initial_guess)
        cp1 = initial_guess[0,:,0].reshape(1,-1)
        cp2 = initial_guess[:,:,-1]
        cps = np.concatenate((cp1,cp2), axis=0).transpose()
        self.progs[0].SetInitialGuess(trajopt.control_points(), cps)
        self.progs[0].SetInitialGuess(trajopt.duration(), 0.1)
    
    def add_task_constraints(self):
        super().add_task_constraints()

        self.ntimings = self.timings / np.max(self.timings)

        for (i,timing) in enumerate(self.ntimings):
            self._add_pose_constraint(self.trajopts[0], self.waypoints[0,i],
                                      self.tol_translation, self.tol_rotation,timing)

    def solve(self):
        self._solve(self.progs[0])

    def add_joint_cp_error_cost(self):
        num_q = self.robot.plant.num_positions()
        for j in range(0, self.num_control_points):
            self.progs[0].AddQuadraticErrorCost(
                self.coeff_joint_cp_error*np.eye(num_q), self.demo.ys[0,j], 
                self.trajopts[0].control_points()[:, j]
            )

    def add_jerk_cost(self):
        bound = np.array([500 for _ in range(7)])
        self.trajopts[0].AddJerkBounds(-bound,bound)
        ts = np.linspace(0, 1, 100)
        cost = 0
        for i in range(1,len(ts)):
            jerk = self.sym_rjerk[0].value(ts[i])
            dt = ts[i] - ts[i-1]
            cost += matmul(jerk.transpose(),jerk)[0,0] * dt * self.coeff_jerk * pow(self.trajopts[0].duration(),6) / (len(ts)*self.robot.plant.num_positions())
            
            # self.progs[0].AddCost(matmul(jerk.transpose(),jerk)[0,0] * dt #* self.trajopts[0].duration()
            #                        * self.coeff_jerk / (len(ts)*self.robot.plant.num_positions()))

        self.jerk_cost = self.progs[0].AddCost(cost)

        # cps = self.sym_rjerk[0].control_points()
        # for j in range(len(cps)):
        #     self.progs[0].AddCost(matmul(cps[j].transpose(),cps[j])[0,0] * self.coeff_jerk / (len(cps)*self.num_segments*self.robot.plant.num_positions()))

    def add_vel_cost(self):
        ts = np.linspace(0, 1, 100)
        cost = 0
        limit = self.trajopts[0].duration()* self.robot.plant.GetVelocityUpperLimits()
        for i in range(1,len(ts)):
            vel = self.sym_rvel[0].value(ts[i])
            # obj = matmul(limit.transpose(),limit) - pow(vel,2)
            dt = ts[i] - ts[i-1]
            cost += matmul((limit-vel).transpose(),(limit+vel))[0,0] * dt * self.coeff_vel / (len(ts)*self.robot.plant.num_positions())

        self.vel_cost = self.progs[0].AddCost(cost)
#%%
    
def read_data():
    with open('/home/abrk/catkin_ws/src/lfd/lfd_dmp/scripts/experimental/smoothing/00.pickle', 'rb') as file:
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
demo = Demonstration(ts, ys, positions, orientations)
demo.filter(thr_translation=thr_translation)

robot = FR3Drake(franka_pkg_xml="/home/abrk/thesis/drake/franka_drake/package.xml",
                 urdf_path="package://franka_drake/urdf/fr3_nohand.urdf")

config_1 = {
    "num_cps" : 4,
    "bspline_order" : 4,
    "wp_per_segment" : 2,
    "overlap" : 1,
    "bound_duration" : [0.01,5],
    "coeff_duration" : 1,
    "coeff_joint_cp_error" : 1,
    "solver_log" : "/tmp/trajopt1.txt",
}

config_ip = {
    "num_cps" : demo.length,
    "bspline_order" : 4,
    "wp_per_segment" : demo.length,
    "overlap" : 0,
    "bound_duration" : [0.01,5],
    "coeff_duration" : 1,
    "coeff_jerk" : 0.04,
    "coeff_vel" : 0.0,
    "coeff_joint_cp_error" : 1,
    "tol_translation" : 0.02,
    "tol_rotation" : 0.1,
    "solver_log" : "/tmp/trajopt2.txt",
    "solver" : IpoptSolver()
}


smoother = TrajectoryOptimizer(robot, config_1)
smoother.input_demo(demo)
smoother.init_trajopts()
smoother.add_pos_vel_bounds()
smoother.add_duration_bound()
smoother.add_duration_cost()
smoother.add_joint_constraints()
smoother.finalize()
smoother.solve()
smoother.plot_trajectory(smoother.compile_trajectory())
initial_guess = smoother.export_initial_guess()
timings = smoother.export_timing()


#%%
smoother = SingleOptimizer(robot, config_ip)
smoother.input_demo(demo)
smoother.init_trajopts()
smoother.add_pos_vel_bounds()
smoother.add_duration_bound()
smoother.add_duration_cost()
smoother.import_timing(timings)
smoother.add_task_constraints()
smoother.add_joint_cp_error_cost()
smoother.add_jerk_cost()
smoother.add_vel_cost()
smoother.apply_initial_guess(initial_guess)
smoother.solve()
smoother.plot_trajectory(smoother.compile_trajectory())

# guess = smoother.trajopts[0].ReconstructTrajectory(smoother.result)



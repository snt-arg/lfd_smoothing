import numpy as np
import matplotlib.pyplot as plt
import copy

from pydrake.all import *
from manipulation.meshcat_utils import PublishPositionTrajectory

class TrajectorySmoother:

    def __init__(self, robot, config):
        self.robot = robot
        self.config = config
        # self.num_control_points = config["num_cps"]
        # self.bspline_order = config["bspline_order"]
        # self.demo = config.get("demo", None)
        # self.wp_per_segment = config.get("wp_per_segment", self.num_control_points)
        # self.overlap = config.get("overlap", 0)
        
        # self.vel_bound = config.get("bound_velocity", None)
        # self.acc_bound = config.get("bound_acceleration", None)
        # self.jerk_bound = config.get("bound_jerk", None)
        # self.duration_bound = config.get("bound_duration", None)

        # self.coeff_duration = config.get("coeff_duration", None)
        # self.coeff_jerk = config.get("coeff_jerk", None)
        # self.coeff_joint_cp_error = config.get("coeff_joint_cp_error", None)
        # self.coeff_vel = config.get("coeff_vel", None)

        # self.tol_joint = config.get("tol_joint", None)
        # self.tol_translation = config.get("tol_translation", None)
        # self.tol_rotation = config.get("tol_rotation", None)

        # self.init_guess_cps = config.get("init_guess_cps", None)
        # self.waypoints_ts = config.get("waypoints_ts", None)

        # self.doplot = config.get("plot", True)

        # self.solver = config.get("solver", None)
        # self.solver_log = config.get("solver_log", "/tmp/trajopt.txt")

    def run(self):
        if self.config.demo is not None: self.input_demo()
        self.init_trajopts()
        self.add_pos_bounds()
        if self.config.waypoints_ts is not None: self.import_waypoints_ts(self.config.waypoints_ts)
        if self.config.vel_bound is not None: self.add_vel_bounds()
        if self.config.acc_bound is not None: self.add_acc_bounds()
        if self.config.jerk_bound is not None: self.add_jerk_bounds()
        if self.config.duration_bound is not None: self.add_duration_bound()
        if self.config.coeff_duration is not None: self.add_duration_cost()
        if self.config.tol_joint is not None: self.add_joint_constraints()
        if self.config.tol_translation is not None and self.config.tol_rotation is not None: self.add_task_constraints()
        if self.config.coeff_jerk is not None: self.add_jerk_cost()
        if self.config.coeff_joint_cp_error is not None: self.add_joint_cp_error_cost()
        if self.config.coeff_vel is not None: self.add_vel_cost()
        if self.num_segments > 1: self.join_trajopts()
        if self.config.init_guess_cps is not None: self.set_init_guess_cps(self.config.init_guess_cps)
        self.solve()
        result_traj = self.compile_trajectory()
        if self.config.doplot is True: self.plot_trajectory(result_traj)



    def input_demo(self):

        self.demo = copy.deepcopy(self.config.demo)
        self.demo.divisible_by(self.config.wp_per_segment, self.config.overlap)
        self.waypoints = []
        self.set_waypoints(self.demo)
        self._segment_demo()
    
    def _segment_demo(self):
        step = self.config.wp_per_segment
        self.demo.reshape(step, self.config.overlap)
        self.waypoints = self.demo.split_into_segments(np.array(self.waypoints), 
                                                  step, self.config.overlap)
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
                              for i in range(self.config.num_control_points)]
        basis_exp = BsplineBasis_[Expression](trajopt.basis().order(), 
                                              trajopt.basis().knots())  
        return BsplineTrajectory_[Expression](basis_exp,control_points_exp)
    
    def add_pos_bounds(self):
        for trajopt in self.trajopts:
            trajopt.AddPositionBounds(
                self.robot.plant.GetPositionLowerLimits(), 
                self.robot.plant.GetPositionUpperLimits()
            )
    def add_vel_bounds(self):
        for trajopt in self.trajopts:
            trajopt.AddVelocityBounds(
                self.config.vel_bound[0],
                self.config.vel_bound[1]
            )
    def add_acc_bounds(self):
        for trajopt in self.trajopts:
            trajopt.AddAccelerationBounds(
                self.config.acc_bound[0],
                self.config.acc_bound[1]
            )
    def add_jerk_bounds(self):
        for trajopt in self.trajopts:
            trajopt.AddJerkBounds(
                self.config.jerk_bound[0],
                self.config.jerk_bound[1]
            )

    def add_duration_bound(self):
        for trajopt in self.trajopts:
            trajopt.AddDurationConstraint(self.config.duration_bound[0],
                                          self.config.duration_bound[1])


    def add_duration_cost(self):
        for trajopt in self.trajopts:
            trajopt.AddDurationCost(self.config.coeff_duration)


    def init_trajopts(self):
        self.trajopts = []
        self.progs = []
             
        for _ in range(self.num_segments):
            trajopt = KinematicTrajectoryOptimization(self.robot.plant.num_positions(), 
                                                    self.config.num_control_points,
                                                    spline_order=self.config.bspline_order)
            self.trajopts.append(trajopt)
            self.progs.append(trajopt.get_mutable_prog())
        
        self._make_symbolic()

    def add_jerk_cost(self):
        # This function would only be correct if num_cps==bspline_order==4
        for i in range(self.num_segments):
            cps = self.sym_rjerk[i].control_points()
            for j in range(len(cps)):
                self.progs[i].AddCost(matmul(cps[j].transpose(),cps[j])[0,0] * self.config.coeff_jerk 
                                      / pow(self.trajopts[i].duration(),5) # Order is one less than expected (6) because of Jdt
                                      / (len(cps)*self.num_segments*self.robot.plant.num_positions()))
    
    
    def _add_joint_constraint(self, trajopt, position, tolerance, ntime, rest=False):
        trajopt.AddPathPositionConstraint(position - tolerance, position + tolerance, ntime)
        if rest:
            self._add_zerovel_constraint(trajopt,ntime)

    def add_joint_constraints(self):

        self._add_joint_constraint(self.trajopts[0], self.demo.ys[0,0], 0, 0, rest=True)
        self._add_joint_constraint(self.trajopts[-1], self.demo.ys[-1,-1], 0, 1, rest=True)

        for i in range(0,self.num_segments - 1):
            self._add_joint_constraint(self.trajopts[i], self.demo.ys[i,-1], self.config.tol_joint, 1)
    
    def add_joint_cp_error_cost(self):
        num_q = self.robot.plant.num_positions()
        if self.wp_per_segment == self.num_control_points:
            for i in range(0,self.num_segments):
                for j in range(1, self.num_control_points):
                    self.progs[i].AddQuadraticErrorCost(
                        self.config.coeff_joint_cp_error*np.eye(num_q), self.demo.ys[i,j], self.trajopts[i].control_points()[:, j]
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
                                      self.config.tol_translation, self.config.tol_rotation, 1)
    
    def add_vel_cost(self):
        raise NotImplementedError()
    
    def join_trajopts(self):
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
        solver_options.SetOption(CommonSolverOption.kPrintFileName, self.config.solver_log)

        if self.config.solver is None:
            self.result = Solve(prog, solver_options=solver_options)
        else:
            self.result = self.config.solver.Solve(prog, solver_options=solver_options)
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
            [result.GetSolution(trajopt.control_points())[:,i,np.newaxis] for i in range(self.config.num_control_points)])
    
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
    
    def export_cps(self):
        cps = []
        for trajopt in self.trajopts:
            cps.append(self.result.GetSolution(trajopt.control_points()))
        return cps
    
    def set_init_guess_cps(self, initial_guess):
        for (i,trajopt) in enumerate(self.trajopts):
            self.nprog.SetInitialGuess(trajopt.control_points(), initial_guess[i])

    def export_waypoint_ts(self):
        timings = [0]
        for trajopt in self.trajopts:
            timings.append(timings[-1] + self.result.GetSolution(trajopt.duration()))
        return timings
    
    def import_waypoints_ts(self, timings):
        self.timings = timings

import numpy as np

from pydrake.all import *
from lfd_smoother.core.trajectory_optimizer import TrajectoryOptimizer

class SingleOptimizer(TrajectoryOptimizer):

    def __init__(self, robot, config):
        super().__init__(robot, config)

        
    def set_init_guess_cps(self, initial_guess):
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
                                      self.config.tol_translation, self.config.tol_rotation,timing)
    
    def add_duration_bound(self):
        duration = np.max(self.timings)  
        trajopt = self.trajopts[0]
        trajopt.AddDurationConstraint(max(0,duration + self.config.duration_bound[0]),
                                        duration + self.config.duration_bound[1])

    def solve(self):
        self._solve(self.progs[0])
    
    def _solve(self,prog):
        solver_options = SolverOptions()
        solver_options.SetOption(IpoptSolver.id(), "tol", 1e-6)
        solver_options.SetOption(IpoptSolver.id(), "max_iter", 5000)
        solver_options.SetOption(CommonSolverOption.kPrintFileName, self.config.solver_log)
        # solver_options.SetOption(IpoptSolver.id(), "print_level", 5)
        prog.SetSolverOptions(solver_options)

        solver = IpoptSolver()
        self.result = solver.Solve(prog)
        self.success = self.result.is_success()
        
        if not self.success:
            print("Optimization Failed")
        else:
            print("Optimization Finished Successfully")


    def add_joint_cp_error_cost(self):
        num_q = self.robot.plant.num_positions()
        for j in range(0, self.config.num_control_points):
            self.progs[0].AddQuadraticErrorCost(
                self.config.coeff_joint_cp_error*np.eye(num_q), self.demo.ys[0,j], 
                self.trajopts[0].control_points()[:, j]
            )

    def add_jerk_cost(self):
        ts = np.linspace(0, 1, 100)
        cost = 0
        for i in range(1,len(ts)):
            jerk = self.sym_rjerk[0].value(ts[i])
            dt = ts[i] - ts[i-1]
            cost += matmul(jerk.transpose(),jerk)[0,0] * dt * self.config.coeff_jerk * pow(self.trajopts[0].duration(),6) / (len(ts)*self.robot.plant.num_positions())
            
        self.jerk_cost = self.progs[0].AddCost(cost)

    def add_vel_cost(self):
        ts = np.linspace(0, 1, 100)
        cost = 0
        limit = self.trajopts[0].duration()* self.robot.plant.GetVelocityUpperLimits()
        for i in range(1,len(ts)):
            vel = self.sym_rvel[0].value(ts[i])
            dt = ts[i] - ts[i-1]
            cost += matmul((limit-vel).transpose(),(limit+vel))[0,0] * dt * self.config.coeff_vel / (len(ts)*self.robot.plant.num_positions())

        self.vel_cost = self.progs[0].AddCost(cost)
        
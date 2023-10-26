import numpy as np
import yaml

class RobotConfig:

    def __init__(self, robot, demo) -> None:
        self.ref_vel_bound = np.array([robot.plant.GetVelocityLowerLimits(),
                    robot.plant.GetVelocityUpperLimits()
                    ])

        self.demo = demo
    
    def parse_from_file(self,filename):
        with open(filename, "r") as file:
            yaml_data = yaml.safe_load(file)
        
        self.parse(yaml_data)    

    def parse(self,config : dict):
        self.num_control_points = config.get("num_cps",self.demo.length)
        self.bspline_order = config.get("bspline_order", 4)
        self.wp_per_segment = config.get("wp_per_segment", self.num_control_points)
        self.overlap = config.get("overlap", 0)

        coeff = config.get("velocity_scaling", None)
        if coeff is not None: self.vel_bound = float(coeff) * self.ref_vel_bound
        else: self.vel_bound = None

        coeff = config.get("acceleration_scaling", None)
        if coeff is not None: self.acc_bound = float(coeff) * self.ref_acc_bound
        else: self.acc_bound = None

        coeff = config.get("jerk_scaling", None)
        if coeff is not None: self.jerk_bound = float(coeff) * self.ref_jerk_bound
        else: self.jerk_bound = None

        self.duration_bound = config.get("duration_bound", None)

        self.coeff_duration = config.get("coeff_duration", None)
        self.coeff_jerk = config.get("coeff_jerk", None)
        self.coeff_joint_cp_error = config.get("coeff_joint_cp_error", None)
        self.coeff_vel = config.get("coeff_vel", None)

        self.tol_joint = config.get("tol_joint", None)
        self.tol_translation = config.get("tol_translation", None)
        self.tol_rotation = config.get("tol_rotation", None)

        self.doplot = config.get("plot", True)
        self.solver_log = config.get("solver_log", "/tmp/trajopt.txt")  

        self.init_guess_cps = None
        self.waypoints_ts = None
        self.solver = None              

    def add_initial_guess(self, init_guess_cps=None, waypoints_ts=None):
        self.init_guess_cps = init_guess_cps
        self.waypoints_ts = waypoints_ts

    def add_solver(self, solver):
        self.solver = solver


class FR3Config(RobotConfig):

    def __init__(self, robot, demo) -> None:
        super().__init__(robot, demo)

        self.ref_acc_bound = np.array([[-10.0 for _ in range(robot.plant.num_positions())],
                            [10.0 for _ in range(robot.plant.num_positions())]
                            ])
        self.ref_jerk_bound = np.array([[-500.0 for _ in range(robot.plant.num_positions())],
                    [500.0 for _ in range(robot.plant.num_positions())]
                            ])

class YumiConfig(RobotConfig):

    def __init__(self, robot, demo) -> None:
        super().__init__(robot, demo)

        self.ref_acc_bound = np.array([[-10.0 for _ in range(robot.plant.num_positions())],
                            [10.0 for _ in range(robot.plant.num_positions())]
                            ])
        self.ref_jerk_bound = np.array([[-500.0 for _ in range(robot.plant.num_positions())],
                    [500.0 for _ in range(robot.plant.num_positions())]
                            ])        

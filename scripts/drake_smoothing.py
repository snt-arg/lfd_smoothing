#!/usr/bin/env python3

from lfd_smoother.core.smoother import TrajectorySmoother
from lfd_smoother.core.single_smoother import SingleSmoother
from lfd_smoother.util.demonstration import Demonstration
from lfd_smoother.util.fr3_drake import FR3Drake

from pydrake.all import *

thr_translation = 0.01

demo = Demonstration()
demo.read_from_json(filename='demo_samples/json/00.json')
demo.filter(thr_translation=thr_translation)

robot = FR3Drake(franka_pkg_xml="/home/abrk/thesis/drake/franka_drake/package.xml",
                 urdf_path="package://franka_drake/urdf/fr3_nohand.urdf")

config_1 = {
    "num_cps" : 4,
    "bspline_order" : 4,
    "wp_per_segment" : 2,
    "overlap" : 1,
    "demo" : demo,
    "bound_velocity" : [robot.plant.GetVelocityLowerLimits(),
                    robot.plant.GetVelocityUpperLimits()
                    ],
    "bound_duration" : [0.01,5],
    "coeff_duration" : 1,
    "tol_joint" : 0,
    "solver_log" : "/tmp/trajopt1.txt",
    "plot" :  True
}


smoother = TrajectorySmoother(robot, config_1)
smoother.run()
initial_guess = smoother.export_cps()
timings = smoother.export_waypoint_ts()


config_ip = {
    "num_cps" : demo.length,
    "bspline_order" : 4,
    "wp_per_segment" : demo.length,
    "overlap" : 0,
    "demo" : demo,
    "bound_velocity" : [robot.plant.GetVelocityLowerLimits(),
                        robot.plant.GetVelocityUpperLimits()
                        ],
    "bound_acceleration" : [[-10 for _ in range(robot.plant.num_positions())],
                            [10 for _ in range(robot.plant.num_positions())]
                            ],
    "bound_jerk" : [[-500 for _ in range(robot.plant.num_positions())],
                    [500 for _ in range(robot.plant.num_positions())]
                            ],                       
    "bound_duration" : [0.01,5],
    "coeff_duration" : 1,
    "coeff_jerk" : 0.04,
    # "coeff_vel" : 0.0,
    "coeff_joint_cp_error" : 1,
    "tol_translation" : 0.02,
    "tol_rotation" : 0.1,
    "solver_log" : "/tmp/trajopt2.txt",
    "solver" : IpoptSolver(),
    "init_guess_cps" : initial_guess,
    "waypoints_ts" : timings,
    "plot" : True
}

smoother = SingleSmoother(robot, config_ip)
smoother.run()



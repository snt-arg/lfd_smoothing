#!/usr/bin/env python3

import yaml

from pydrake.all import *

from lfd_smoother.core.smoother import TrajectorySmoother
from lfd_smoother.core.single_smoother import SingleSmoother
from lfd_smoother.util.demonstration import Demonstration
from lfd_smoother.util.fr3_drake import FR3Drake
from lfd_smoother.util.config import FR3Config

thr_translation = 0.01

demo = Demonstration()
demo.read_from_json(filename='demo_samples/json/00.json')
demo.filter(thr_translation=thr_translation)

robot = FR3Drake(franka_pkg_xml="drake/franka_drake/package.xml",
                 urdf_path="package://franka_drake/urdf/fr3_nohand.urdf")

config = FR3Config(robot, demo)
config.parse_from_file("config/initial.yaml")



smoother = TrajectorySmoother(robot, config)
smoother.run()
initial_guess = smoother.export_cps()
timings = smoother.export_waypoint_ts()

config.parse_from_file("config/main.yaml")
config.add_initial_guess(initial_guess,timings)
config.add_solver(IpoptSolver())

smoother = SingleSmoother(robot, config)
smoother.run()



#!/usr/bin/env python3

from pydrake.all import *

from lfd_smoother.core.trajectory_optimizer import TrajectoryOptimizer
from lfd_smoother.core.single_optimizer import SingleOptimizer
from lfd_smoother.util.demonstration import Demonstration
from lfd_smoother.util.fr3_drake import FR3Drake
from lfd_smoother.util.config import FR3Config

from lfd_smoother.api.trajectory_smoother import TrajectorySmoother

smoother_config= {
"demo_filter_threshold": 0.01,
"franka_pkg_xml": "drake/franka_drake/package.xml",
"urdf_path": "package://franka_drake/urdf/fr3_nohand.urdf",
"config_hierarchy": ["config/opt/initial.yaml", "config/opt/main.yaml"]
}

smoother = TrajectorySmoother(smoother_config)


smoother.read_demo_json('demo_samples/json/00.json')
smoother.run()



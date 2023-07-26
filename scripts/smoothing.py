#!/usr/bin/env python3

from pydrake.all import *

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


#%%

trajopt = smoother.smoother.trajopts[0]

traj = trajopt.ReconstructTrajectory(smoother.smoother.result)

smoother.smoother.plot_trajectory(traj)

#%% 

import pickle

serialized_trajectory = pickle.dumps(traj)


newtraj = pickle.loads(serialized_trajectory)

smoother.smoother.plot_trajectory(newtraj)

#%% normalize trajectory

duration = traj.end_time() - traj.start_time()

basis = traj.basis()
scaled_knots = [(knot / duration) for knot in basis.knots()]

ntraj = BsplineTrajectory(
    BsplineBasis(basis.order(), scaled_knots),
    traj.control_points())

smoother.smoother.plot_trajectory(ntraj)

#%% Store JSON




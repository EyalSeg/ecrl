from toolz import curry
from numpy import append

from algorithms.algorithm_typing import Trajectory


@curry
def last_observation_bc(traj: Trajectory, add_timestep=False):
    last_observation = traj.observations[-1]

    if add_timestep:
        last_observation = append(last_observation, len(traj.observations))

    return last_observation
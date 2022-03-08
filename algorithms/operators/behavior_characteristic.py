from toolz import curry
from numpy import append

from algorithms.algorithm_typing import Trajectory


@curry
def last_observation_bc(env, traj: Trajectory, add_timestep=False):
    last_observation = traj.observations[-1]

    if add_timestep:
        last_observation = append(last_observation, len(traj.observations) / env._max_episode_steps)

    return last_observation
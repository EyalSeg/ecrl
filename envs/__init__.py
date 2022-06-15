import gym
from mujoco_maze import PointEnv, AntEnv

from envs.large_maze_deceptive import LargeMazeDeceptive
from envs.small_maze_deceptive import SmallMazeDeceptive

gym.envs.register(
    id="PointMazeSmallDeceptive-v0",
    entry_point="envs.step_limited_maze_env:StepLimitedMazeEnv",
    kwargs=dict(
        model_cls=PointEnv,
        maze_task=SmallMazeDeceptive,
        maze_size_scaling=SmallMazeDeceptive.MAZE_SIZE_SCALING.point,
        inner_reward_scaling=0,
        time_limit=200
    )
)

gym.envs.register(
    id="AntMazeSmallDeceptive-v0",
    entry_point="envs.step_limited_maze_env:StepLimitedMazeEnv",
    kwargs=dict(
        model_cls=AntEnv,
        maze_task=SmallMazeDeceptive,
        maze_size_scaling=SmallMazeDeceptive.MAZE_SIZE_SCALING.ant,
        inner_reward_scaling=0,
        time_limit=500
    )
)

gym.envs.register(
    id="PointMazeLargeDeceptive-v0",
    entry_point="envs.step_limited_maze_env:StepLimitedMazeEnv",
    kwargs=dict(
        model_cls=PointEnv,
        maze_task=LargeMazeDeceptive,
        maze_size_scaling=LargeMazeDeceptive.MAZE_SIZE_SCALING.point,
        inner_reward_scaling=0,
        time_limit=2000,
    )
)

gym.envs.register(
    id="AntMazeLargeDeceptive-v0",
    entry_point="envs.step_limited_maze_env:StepLimitedMazeEnv",
    kwargs=dict(
        model_cls=AntEnv,
        maze_task=LargeMazeDeceptive,
        maze_size_scaling=LargeMazeDeceptive.MAZE_SIZE_SCALING.ant,
        inner_reward_scaling=0,
        time_limit=5000
    )
)
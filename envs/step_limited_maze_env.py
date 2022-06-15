from mujoco_maze.maze_env import MazeEnv


class StepLimitedMazeEnv(MazeEnv):
    def __init__(self, time_limit, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._max_episode_steps = time_limit

    def step(self, action):
        next_obs, reward, done, info = super().step(action)

        if self.t >= self._max_episode_steps:
            done = True

        return next_obs, reward, done, info

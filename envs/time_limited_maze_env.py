from mujoco_maze.maze_env import MazeEnv


class TimeLimitedMazeEnv(MazeEnv):
    def __init__(self, time_limit, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_limit = time_limit

    def step(self, action):
        next_obs, reward, done, info = super().step(action)

        if self.t >= self.time_limit:
            done = True

        return next_obs, reward, done, info

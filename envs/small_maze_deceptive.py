import numpy as np
from mujoco_maze.maze_env_utils import MazeCell
from mujoco_maze.maze_task import MazeTask, MazeGoal


class SmallMazeDeceptive(MazeTask):
    REWARD_THRESHOLD: float = 0
    PENALTY: float = 0

    def __init__(self, scale):
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, -6.0]) * scale, threshold=1)]

    def reward(self, obs):
        if self.termination(obs):
            return 10000

        distances = [goal.euc_dist(obs) for goal in self.goals]

        return sum(distances) * -1

    @staticmethod
    def create_maze():
        o, I, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT,
        return [
            [I, I, I, I, I, I, I],
            [I, o, o, o, o, o, I],
            [I, o, o, o, o, o, I],
            [I, o, o, o, o, o, I],
            [I, o, o, o, o, o, I],
            [I, o, I, I, I, o, I],
            [I, o, I, o, I, o, I],
            [I, o, I, R, I, o, I],
            [I, o, o, o, o, o, I],
            [I, I, I, I, I, I, I],

        ]
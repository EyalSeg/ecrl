from functools import partial

import toolz

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.operators import truncated_selection
from experiments.scripts.loggers.composite_logger import CompositeLogger
from experiments.scripts.loggers.console_logger import ConsoleLogger
from experiments.scripts.loggers.wandb_log import WandbLogger
from experiments.scripts.trainer import Trainer


if __name__ == "__main__":
    env_name = "Acrobot-v1"
    validation_episodes = 100
    fit_robustness = 5
    mutation_strength = 0.003
    truncation_size = 10
    population_size = 50

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Genetic Algorithm",
            "Environment": env_name,
            "Validation Episodes": validation_episodes,
            "Fitness Robustness": fit_robustness,
            "Mutation Strength": mutation_strength,
            "Truncation Size": truncation_size,
            "Population Size": 50
        })
    ])

    trainer = Trainer(env_name=env_name, max_train_steps=int(1e6), validation_episodes=validation_episodes, logger=logger)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   256,
                   512,
                   trainer.train_env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    fitness = trainer.episodic_rewards(trainer.train_env, n_episodes=fit_robustness)
    mutator = add_gaussian_noise(mutation_strength)
    selector = truncated_selection(truncation_size)

    ga = GeneticAlgorithm(
        pop_size=population_size,
        initializer=initializer,
        fitness=fitness,
        mutator=mutator,
        survivors_selector=selector
    )

    trainer.fit(ga)


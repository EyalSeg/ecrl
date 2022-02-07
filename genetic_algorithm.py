from argparse import ArgumentParser
from functools import partial

import toolz

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.operators import truncated_selection
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from loggers.wandb_log import WandbLogger
from algorithms.trainer import Trainer


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("env", type=str)
    parser.add_argument("popsize", type=int)
    parser.add_argument("validation_episodes", type=int)
    parser.add_argument("fitness_robustness", type=int)
    parser.add_argument("mutation_strength", type=float)
    parser.add_argument("truncation_size", type=int)

    args = parser.parse_args()

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Genetic Algorithm",
            "Environment": args.env,
            "Population Size": args.popsize,
            "Validation Episodes": args.validation_episodes,
            "Fitness Robustness": args.fitness_robustness,
            "Mutation Strength": args.mutation_strength,
            "Truncation Size": args.truncation_size,
        })
    ])

    trainer = Trainer(env_name=args.env,
                      max_train_steps=int(1e6),
                      validation_episodes=args.validation_episodes,
                      logger=logger)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   256,
                   512,
                   trainer.train_env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    fitness = trainer.episodic_rewards(trainer.train_env, n_episodes=args.robustness)
    mutator = add_gaussian_noise(args.mutation_strength)
    selector = truncated_selection(args.truncation_size)

    ga = GeneticAlgorithm(
        pop_size=args.popsize,
        initializer=initializer,
        fitness=fitness,
        mutator=mutator,
        survivors_selector=selector
    )

    trainer.fit(ga)


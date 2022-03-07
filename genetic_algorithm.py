from argparse import ArgumentParser
from functools import partial

import toolz

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.operators.selection import truncated_selection, find_true_elite
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from loggers.wandb_log import WandbLogger
from algorithms.trainer import Trainer


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--popsize", type=int, required=True)
    parser.add_argument("--validation_episodes", type=int, required=True)
    parser.add_argument("--elite_robustness", type=int, required=True)
    parser.add_argument("--elite_candidates", type=int, required=True)
    parser.add_argument("--mutation_strength", type=float, required=True)
    parser.add_argument("--truncation_size", type=int, required=True)
    parser.add_argument("--train_steps", type=int, default=int(1e6))
    parser.add_argument("--elitism", type=int, default=1,)

    args = parser.parse_args()

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Genetic Algorithm",
            "env": args.env,
            "popsize": args.popsize,
            "validation_episodes": args.validation_episodes,
            "elite_robustness": args.elite_robustness,
            "elite_candidates": args.elite_candidates,
            "mutation_strength": args.mutation_strength,
            "truncation_size": args.truncation_size,
            "elitism": args.elitism
        })
    ])

    trainer = Trainer(env_name=args.env,
                      max_train_steps=args.train_steps,
                      validation_episodes=args.validation_episodes,
                      logger=logger)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   56,
                   56,
                   56,
                   trainer.train_env.action_space.n]

    initializer = partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims)
    fitness = trainer.episodic_rewards(trainer.train_env, n_episodes=1)
    mutator = add_gaussian_noise(args.mutation_strength)
    selector = truncated_selection(args.truncation_size)
    elite_extractor = find_true_elite(args.elite_candidates, fitness, args.elite_robustness)

    ga = GeneticAlgorithm(
        pop_size=args.popsize,
        initializer=initializer,
        fitness=fitness,
        mutator=mutator,
        survivors_selector=selector,
        elite_extractor=elite_extractor,
        elitism=args.elitism
    )

    trainer.fit(ga)


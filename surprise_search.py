from argparse import ArgumentParser
from functools import partial

import toolz

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from algorithms.surprise_search import SurpriseSearch

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

import pytorch_lightning as pl

from agents.pytorch import LinearTorchPolicy, TorchPolicyAgent, add_gaussian_noise
from algorithms.operators.selection import truncated_selection
from algorithms.trainer import Trainer
from loggers.composite_logger import CompositeLogger
from loggers.console_logger import ConsoleLogger
from loggers.wandb_log import WandbLogger


class BehaviourLearner(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr):
        super(BehaviourLearner, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

        self.net = torch.nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, X):
        return self.net(X)

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch

        loss = self.get_loss(X, y)

        return {"loss": loss}

    def validation_step(self, validation_batch, batch_idx):
        X, y = validation_batch

        loss = self.get_loss(X, y)

        return {"loss": loss}

    def get_loss(self, X, y):
        pred = self.forward(X)
        loss = self.loss_fn(pred, y.squeeze(-1).type(torch.LongTensor))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

@toolz.curry
def measure_surprise(behavior_learner, observations, actions, rewards):
    X = torch.Tensor(observations)
    y = torch.Tensor(actions)

    loss = behavior_learner.get_loss(X, y)
    surprise = loss.item()

    return surprise


@toolz.curry
def train_learner(behavior_model, epochs, train_data, validate_data):
    train_data = TensorDataset(
        torch.Tensor(train_data['observations']),
        torch.Tensor(train_data['actions'])
    )

    validate_data = TensorDataset(
        torch.Tensor(validate_data['observations']),
        torch.Tensor(validate_data['actions'])
    )

    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True, num_workers=4)
    validate_loader = DataLoader(validate_data, batch_size=1000, shuffle=False, num_workers=4)

    trainer = pl.Trainer(max_epochs=epochs)

    trainer.fit(behavior_model, train_loader, validate_loader)

    # TODO: check if it validates, add short stopping


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--validation_episodes", type=int)
    parser.add_argument("--mutation_strength", type=float)
    parser.add_argument("--truncation_size", type=int)
    parser.add_argument("--train_steps", type=int, default=int(1e6))
    parser.add_argument("--behavior_learner_epochs", type=int)
    parser.add_argument("--behavior_lr", type=float)
    parser.add_argument("--replay_buffer_size", type=int)

    args = parser.parse_args()

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Surprise Search",
            "Environment": args.env,
            "Population Size": args.popsize,
            "Validation Episodes": args.validation_episodes,
            "Mutation Strength": args.mutation_strength,
            "Truncation Size": args.truncation_size,
            "Behavior Learner Epochs": args.behavior_learner_epochs,
            "Behavior Learning Rate": args.behavior_lr,
            "Replay Buffer Size": args.replay_buffer_size
        })
    ])

    trainer = Trainer(env_name="Acrobot-v1",
                      max_train_steps=args.train_steps,
                      validation_episodes=args.validation_episodes,
                      logger=logger)

    behavior_learner = BehaviourLearner(
        input_dim=sum(trainer.train_env.observation_space.shape),
        output_dim=trainer.train_env.action_space.n,
        lr=args.behavior_lr
    )

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   256,
                   512,
                   trainer.train_env.action_space.n]

    ss = SurpriseSearch(
        popsize=args.popsize,
        initializer=partial(toolz.compose_left(LinearTorchPolicy, TorchPolicyAgent), policy_dims),
        rollout=trainer.rollout(trainer.train_env, log_trajectory=True),
        train_learner=train_learner(behavior_learner, args.behavior_learner_epochs),
        replay_buffer_size=args.replay_buffer_size,
        fitness=lambda observations, actions, rewards: sum(rewards),
        surprise=measure_surprise(behavior_learner),
        survivors_selector=truncated_selection(args.truncation_size),
        mutator=add_gaussian_noise(args.mutation_strength),
    )

    trainer.fit(ss)


import math
from argparse import ArgumentParser
from functools import partial

import toolz
import numpy as np

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
        super().__init__()
        super().to(device)

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.lr = lr

        self.net = torch.nn.Sequential(
            nn.Linear(input_dim, 56),
            nn.ReLU(),
            nn.Linear(56, 56),
            nn.ReLU(),
            nn.Linear(56, output_dim),
            nn.Softmax()
        ).to(self.device)

        self.xavier_init()

    def xavier_init(self):
        for name, param in self.net.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            else:
                bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                param.data.uniform_(-bound, bound)

    def forward(self, X):
        return self.net(X)

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch

        loss = self.get_loss(X, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, validation_batch, batch_idx):
        X, y = validation_batch

        loss = self.get_loss(X, y)
        self.log("val_loss", loss)

    def get_loss(self, X, y):
        pred = self.forward(X)
        loss = self.loss_fn(pred, y.squeeze(-1).type(torch.LongTensor).to(self.device))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters()) #, lr=self.lr)
        return optimizer

@toolz.curry
def measure_surprise(behavior_learner, observations, actions, rewards):
    with torch.no_grad():
        X = torch.Tensor(observations).to(device)
        y = torch.Tensor(actions).to(device)

        loss = behavior_learner.get_loss(X, y)
        surprise = loss.item()

        return surprise


@toolz.curry
def train_learner(behavior_model, epochs, early_stop_patience, train_data, validate_data):
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    train_data = TensorDataset(
        torch.Tensor(train_data['observations']).to(device),
        torch.Tensor(train_data['actions']).to(device)
    )

    validate_data = TensorDataset(
        torch.Tensor(validate_data['observations']).to(device),
        torch.Tensor(validate_data['actions']).to(device)
    )

    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True, num_workers=4)
    validate_loader = DataLoader(validate_data, batch_size=1000, shuffle=False, num_workers=4)

    trainer = pl.Trainer(max_epochs=epochs,
                         log_every_n_steps=1,
                         accelerator="gpu",
                         gpus=1,
                         auto_lr_find=True,
                         callbacks=[
                             EarlyStopping(monitor="val_loss", patience=early_stop_patience)
                         ])

    trainer.tune(behavior_model, train_loader, validate_loader)
    trainer.fit(behavior_model, train_loader, validate_loader)

    behavior_model.to(device)

    # TODO: check if it validates, add short stopping

@toolz.curry
def robust_rollout(k, rollout, agent, average_rewards=True):
    rollouts = [rollout(agent) for _ in range(k)]
    states, actions, rewards = zip(*rollouts)

    states = np.row_stack(states)
    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)

    assert states.shape[0] == len(actions) and len(actions) == len(rewards), \
        "state-action-rewards are not at the same length!"

    return states, actions, rewards / k if average_rewards else rewards


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--env", type=str)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--validation_episodes", type=int)
    parser.add_argument("--mutation_strength", type=float)
    parser.add_argument("--truncation_size", type=int)
    parser.add_argument("--fitness_robustness", type=int)
    parser.add_argument("--train_steps", type=int, default=int(1e6))
    parser.add_argument("--behavior_learner_epochs", type=int)
    parser.add_argument("--behavior_early_stop_patience", type=int)
    parser.add_argument("--behavior_lr", type=float)
    parser.add_argument("--replay_buffer_size", type=int)

    args = parser.parse_args()

    logger = CompositeLogger([
        ConsoleLogger(),
        WandbLogger("ecrl", "eyal-segal", config={
            "Algorithm": "Surprise Search",
            "env": args.env,
            "popsize": args.popsize,
            "validation_episodes": args.validation_episodes,
            "mutation_strength": args.mutation_strength,
            "truncation_size": args.truncation_size,
            "fitness_robustness": args.fitness_robustness,
            "behavior_learner_epochs": args.behavior_learner_epochs,
            "behavior_lr": args.behavior_lr,
            "behavior_early_stop_patience": args.behavior_early_stop_patience,
            "replay_buffer_size": args.replay_buffer_size
        })
    ])

    trainer = Trainer(env_name=args.env,
                      max_train_steps=args.train_steps,
                      validation_episodes=args.validation_episodes,
                      logger=logger,
                      log_callbacks=[
                          lambda ss: {"average_surprise": sum(ss.pop_surprises) / len(ss.pop_surprises)},
                          lambda ss: {"max_surprise": max(ss.pop_surprises)},
                      ])

    behavior_learner = BehaviourLearner(
        input_dim=sum(trainer.train_env.observation_space.shape),
        output_dim=trainer.train_env.action_space.n,
        lr=args.behavior_lr
    ).to(device)

    policy_dims = [sum(trainer.train_env.observation_space.shape),
                   56,
                   56,
                   56,
                   trainer.train_env.action_space.n]

    rollout = trainer.rollout(trainer.train_env, log_trajectory=True)

    ss = SurpriseSearch(
        popsize=args.popsize,
        initializer=partial(
            toolz.compose_left(
                LinearTorchPolicy,
                TorchPolicyAgent),
            policy_dims),

        rollout=robust_rollout(args.fitness_robustness, rollout),
        train_learner=train_learner(behavior_learner, args.behavior_learner_epochs, args.behavior_early_stop_patience),
        replay_buffer_size=args.replay_buffer_size,
        fitness=lambda observations, actions, rewards: sum(rewards),
        surprise=measure_surprise(behavior_learner),
        survivors_selector=truncated_selection(args.truncation_size),
        mutator=add_gaussian_noise(args.mutation_strength),
    )

    trainer.fit(ss)


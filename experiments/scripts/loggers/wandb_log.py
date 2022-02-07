import wandb


class WandbLogger:
    def __init__(self, project, entity):
        wandb.init(project=project, entity=entity)

    def log_config(self, config):
        wandb.config = config

    def log(self, metrics):
        wandb.log(metrics)

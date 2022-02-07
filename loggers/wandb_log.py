import wandb


class WandbLogger:
    def __init__(self, project, entity, config):
        wandb.init(project=project, entity=entity, config=config)

    def log(self, metrics):
        wandb.log(metrics)

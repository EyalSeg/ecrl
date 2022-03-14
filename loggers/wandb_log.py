import wandb


class WandbLogger:
    def __init__(self, project, entity, config, group=None):
        wandb.init(project=project, entity=entity, config=config, group=group)

    def log(self, metrics):
        wandb.log(metrics)

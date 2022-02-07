

class CompositeLogger:
    def __init__(self, loggers):
        self.loggers = loggers

    def log_config(self, config):
        for logger in self.loggers:
            logger.log_config(config)

    def log(self, metrics):
        for logger in self.loggers:
            logger.log(metrics)
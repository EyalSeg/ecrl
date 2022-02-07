

class CompositeLogger:
    def __init__(self, loggers):
        self.loggers = loggers

    def log(self, metrics):
        for logger in self.loggers:
            logger.log(metrics)
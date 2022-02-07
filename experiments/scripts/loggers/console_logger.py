from pprint import pprint


class ConsoleLogger:
    def log_config(self, config):
        pprint(config)

    def log(self, metrics):
        pprint(metrics)
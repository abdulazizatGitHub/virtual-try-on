import logging
import config

class Logger(logging.Logger):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        # Directories are created automatically on import, no need to call ensure_directories

        sh = logging.StreamHandler()
        sh.setFormatter(config.log_config.STREAM_FORMATTER)
        sh.setLevel(config.log_config.LEVEL)
        self.addHandler(sh)

        # Create valid filename by replacing dots with underscores
        safe_name = name.replace('.', '_').replace('/', '_').replace('\\', '_')
        fh = logging.FileHandler(config.path_config.LOGS / f"{safe_name}.log", mode='a')
        fh.setFormatter(config.log_config.FILE_FORMATTER)
        fh.setLevel(config.log_config.LEVEL)
        self.addHandler(fh)

    def turn_on(self):
        self.setLevel(config.log_config.LEVEL)
        for handler in self.handlers:
            handler.setLevel(config.log_config.LEVEL)

    def turn_off(self):
        self.setLevel(logging.CRITICAL + 1)
        for handler in self.handlers:
            handler.setLevel(logging.CRITICAL + 1)

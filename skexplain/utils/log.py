import logging
import rootpath


class Logger(logging.getLoggerClass()):
    """ Initialze log with output to given path """

    def __init__(
        self,
        path=f"{rootpath.detect()}/res/log/output.log",
        level=logging.DEBUG,
    ):
        # Create handlers
        super().__init__(__name__)
        self.setLevel(level)

        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(path)
        stream_handler.setLevel(level)
        file_handler.setLevel(level)

        format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(format)
        file_handler.setFormatter(format)

        # Add handlers to the log
        self.addHandler(stream_handler)
        self.addHandler(file_handler)

    def log(self, *args, level=logging.INFO):
        """ Logs message with log with given level """
        super().log(level, " ".join([str(arg) for arg in args]))

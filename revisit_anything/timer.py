import contextlib
import json
import time

from loguru import logger


class Timer:
    def __init__(self, file_name):
        self.file_name = file_name
        self.timed_funcs = {}

    @contextlib.contextmanager
    def timeit_block(self, func_name):
        start_time = time.time()
        yield
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func_name} took {elapsed_time:.2f} seconds")
        self.timed_funcs[func_name] = elapsed_time

    def save_times(self):
        with open(self.file_name, "w") as f:
            json.dump(self.timed_funcs, f, indent=2)
        logger.info(f"Saved timing information to {self.file_name}")

import os
import time

from chaitorch.utils.reporter import Reporter
from chaitorch.training.trigger import BaseTrigger


class Trainer(object):

    def __init__(self, epochs, updater, stop_trigger, out='result'):
        self.epochs = epochs
        self.updater = updater
        self.trigger = BaseTrigger(stop_trigger)
        self.out = out

        self.reporter = Reporter()
        self.reporter.add_observer('training', updater.model)
        self.extensions = []
        self.total_iter = 0
        self.start_at = None

    @property
    def elapsed_time(self):
        return time.time() - self.start_at

    def extend(self, extension):
        self.extensions.append(extension)
        self.extensions = sorted(self.extensions, key=lambda e: e.priority)

    def run(self):
        try:
            os.makedirs(self.out)
        except OSError:
            pass

        self.start_at = time.time()

        while not self.trigger(self):
        # while self.trigger(self):
            self.observation = {}
            with self.reporter.scope(self.observation):
                self.updater.update()
                self.total_iter += 1
                [entry(self) for entry in self.extensions]
        [entry.finalize() for entry in self.extensions]

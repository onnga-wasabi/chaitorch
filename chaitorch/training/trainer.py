import os
import time

import chaitorch.utils.reporter as reporte_mod
from chaitorch.training.trigger import isTrigger


class Trainer(object):

    def __init__(self, updater, stop_trigger, out='result'):
        self.updater = updater
        self.trigger = isTrigger(stop_trigger)
        self.out = out

        self.reporter = reporte_mod.Reporter()
        if hasattr(updater, 'model'):
            self.reporter.add_observer('training', updater.model)
        else:
            for model in updater.models.values():
                self.reporter.add_observer('training', model)
        self.extensions = []
        self.keys = []
        self.total_iter = 0
        self.start_at = None

    @property
    def elapsed_time(self):
        return time.time() - self.start_at

    def extend(self, extension):
        self.extensions.append(extension)
        self.extensions = sorted(self.extensions, key=lambda e: e.priority)

    def run(self):
        self.keys = list(set(self.keys))
        try:
            os.makedirs(self.out)
        except OSError:
            pass

        self.start_at = time.time()

        while not self.trigger(self):
            self.observation = {}
            with self.reporter.scope(self.observation):
                self.updater.update()
                self.total_iter += 1
                [entry(self) for entry in self.extensions]
        [entry.finalize(self) for entry in self.extensions]

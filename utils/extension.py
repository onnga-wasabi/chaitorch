import os
import json
import tempfile
import shutil
import sys

from utils.reporter import Summarizer


class Extension(object):

    def __init__(self, trigger):
        raise NotImplementedError

    def is_trigger(self, trainer):
        if list(self.trigger.keys())[0] == 'epoch':
            return trainer.updater.iteration >= len(trainer.data_loader)
        elif list(self.trigger.keys())[0] == 'iteration':
            return trainer.total_iter % list(self.trigger.values())[0] == 0

    def extension(self, trainer):
        raise NotImplementedError

    def finalize(self):
        pass


class LogReport(Extension):

    def __init__(self, keys, trigger, log_name='log', _print=True):
        self.keys = keys
        self.trigger = trigger
        self.log_name = log_name
        self._init_summary()
        self.log = []
        self._print = _print
        if self._print:
            print(''.join([f'{key}'.ljust(10) if key == 'epoch' else f'{key}'.ljust(20) for key in self.keys]))

    def __call__(self, trainer):
        observation = trainer.observation

        if self.keys is None:
            self.summarizer.add(observation)
        else:
            self.summarizer.add({k: observation[k] for k in self.keys if k in observation})

        if self.is_trigger(trainer):
            results = self.summarizer.compute_mean()
            results['epoch'] = trainer.updater.epoch
            results['iteration'] = trainer.total_iter
            results['elapsed_time'] = trainer.elapsed_time

            self.log.append(results)

            with tempfile.TemporaryDirectory(dir=trainer.out) as tempd:
                path = os.path.join(tempd, 'log.json')
                with open(path, 'w') as wf:
                    json.dump(self.log, wf, indent=4)

                new_path = os.path.join(trainer.out, self.log_name)
                shutil.move(path, new_path)

            if self._print:
                self.printout()

            self._init_summary()

    def printout(self):
        line = ''.join([f'{self.log[-1][key]:}'.ljust(10) if key == 'epoch' else f'{self.log[-1][key]:.5f}'.ljust(20)
                        for key in self.keys])
        sys.stdout.write(f"\033[2K\033[G{line}\n")
        sys.stdout.flush()

    def _init_summary(self):
        self.summarizer = Summarizer({key: 0 for key in self.keys})


class ProgressBar(Extension):

    def __init__(self, update_interval):
        self.update_interval = update_interval

    def __call__(self, trainer):
        if trainer.total_iter % self.update_interval == 0:
            bweight = 50 / len(trainer.data_loader)
            iteration = trainer.updater.iteration
            s = "#" * int(iteration * bweight) + " " * int(50 - iteration * bweight)
            sys.stdout.write(f"\033[2K\033[G[{s}]")
            sys.stdout.flush()

    def finalize(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

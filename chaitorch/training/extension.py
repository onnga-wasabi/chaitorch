import os
import json
import tempfile
import shutil
import sys
import time
import datetime

import torch

import chaitorch.utils.reporter as report_mod
from chaitorch.training.trigger import isTrigger


class Extension(object):

    priority = 0

    def __init__(self, trigger={'epoch': 1}):
        self.trigger = isTrigger(trigger) if isinstance(trigger, dict) else trigger

    def __call__(self, trainer):
        raise NotImplementedError

    def finalize(self, trainer):
        pass


class LogReport(Extension):

    def __init__(self, keys, trigger, log_name='log', _print=True):
        self.keys = keys
        self.trigger = isTrigger(trigger) if isinstance(trigger, dict) else trigger
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

        if self.trigger(trainer):
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
        self.summarizer = report_mod.Summarizer()


class ProgressBar(Extension):

    priority = -1

    def __init__(self, update_interval):
        self.update_interval = update_interval
        self.previous = time.time()
        self.cost = 0

    def __call__(self, trainer):
        if trainer.total_iter % self.update_interval == 0:
            epoch_len = len(trainer.updater.data_loader)
            overall_len = trainer.trigger.trigger * epoch_len
            total_weight = 50 / overall_len
            epoch_weight = 50 / epoch_len

            iteration = trainer.updater.iteration
            progress = "     Total: [" + "#" * int(trainer.total_iter * total_weight) + " " * int(50 - trainer.total_iter * total_weight) + "]"
            this_epoch = "This Epoch: [" + "#" * int(iteration * epoch_weight) + " " * int(50 - iteration * epoch_weight) + "]"

            elapesd = time.time() - self.previous
            self.previous = time.time()
            self.cost += elapesd
            overall_time = (elapesd / self.update_interval) * overall_len
            predited = max(overall_time - self.cost, 0.0)
            estimated_to_finish = f"Estimated time to finish: {str(datetime.timedelta(seconds=predited)):0>8}"
            sys.stdout.write(f"\033[2K\033[G{progress}\n{this_epoch}\n{estimated_to_finish}\033[1A\033[1A\033[G")
            sys.stdout.flush()

    def finalize(self, trainer):
        sys.stdout.write("\033[2K\n\033[2K")
        sys.stdout.flush()


class ClassifyEvaluater(Extension):

    priority = -1

    def __init__(self, data_loader, trigger={'epoch': 1}, eval_fn=None):
        self.data_loader = data_loader
        self.trigger = isTrigger(trigger) if isinstance(trigger, dict) else trigger
        self.eval_fn = eval_fn

    def __call__(self, trainer):
        if self.trigger(trainer):
            reporter = report_mod.Reporter()
            reporter.add_observer('validation', trainer.updater.model)
            summarizer = report_mod.Summarizer()
            for batch in self.data_loader:
                observation = {}
                with reporter.scope(observation):
                    with torch.no_grad():
                        loss_fn = self.eval_fn or trainer.updater.calc_loss
                        trainer.updater.model.eval()
                        loss_fn(batch)
                        trainer.updater.model.train()
                summarizer.add(observation)
            report_mod.report(summarizer.compute_mean())


class SnapshotModel(Extension):

    priority = 0

    def __init__(self, save_dir, trigger={'epoch': 1}):
        self.save_dir = save_dir
        self.trigger = isTrigger(trigger) if isinstance(trigger, dict) else trigger

    def __call__(self, trainer):
        if self.trigger(trainer):
            save_dir = os.path.join(trainer.out, self.save_dir)
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            torch.save(trainer.updater.model.state_dict(), os.path.join(save_dir, 'snapshot_model.params'))

    def finalize(self, trainer):
        torch.save(trainer.updater.model.state_dict(), os.path.join(trainer.out, self.save_dir, 'latest_model.params'))

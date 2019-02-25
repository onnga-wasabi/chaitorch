import chaitorch.utils.reporter as reporte_mod


class isTrigger(object):

    def __init__(self, trigger):
        self.target, self.trigger = list(trigger.items())[0]

    def __call__(self, trainer):
        if self.target == 'epoch':
            return (trainer.total_iter > 0) & (trainer.total_iter % (len(trainer.updater.data_loader) * self.trigger) == 0)

        elif self.target == 'iteration':
            return (trainer.total_iter % self.tigger == 0)


class BestValueTrigger(object):

    def __init__(self, key, compare, trigger={'epoch': 1}):
        self.key = key
        self.best_value = None
        self.trigger = isTrigger(trigger)
        self._init_summary()
        self.compare = compare

    def __call__(self, trainer):
        observation = trainer.observation
        if self.key in observation.keys():
            self.summarizer.add({self.key: observation[self.key]})

        if self.trigger(trainer):
            results = self.summarizer.compute_mean()
            value = results[self.key]
            if self.best_value is None or self.compare(self.best_value, value):
                self.best_value = value
                return True

        return False

    def _init_summary(self):
        self.summarizer = reporte_mod.Summarizer()


class MaxValueTrigger(BestValueTrigger):

    def __init__(self, key, trigger={'epoch': 1}):
        super(MaxValueTrigger, self).__init__(
            key, lambda max_value, new_value: new_value > max_value, trigger)


class MinValueTrigger(BestValueTrigger):

    def __init__(self, key, trigger={'epoch': 1}):
        super(MinValueTrigger, self).__init__(
            key, lambda min_value, new_value: new_value < min_value, trigger)

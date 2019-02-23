class BaseTrigger(object):

    def __init__(self, trigger):
        self.target, self.trigger = list(trigger.items())[0]

    def __call__(self, trainer):
        if self.target == 'epoch':
            return (trainer.total_iter > 0) & (trainer.total_iter % (len(trainer.updater.data_loader) * self.trigger) == 0)

        elif self.target == 'iteration':
            return (trainer.total_iter % self.tigger == 0)

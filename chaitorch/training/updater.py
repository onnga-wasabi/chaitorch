import torch

import chaitorch.utils.reporter as report_mod

OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
}


class Updater(object):

    def __init__(self, model, data_loader, loss_fn=None, device='cpu', compute_accuracy=True, **kwargs):

        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device
        self.compute_accuracy = compute_accuracy

        self.epoch = 0
        self.iteration = 0

        self.set_optimizer(**kwargs)

    def set_optimizer(self, **kwargs):
        self.optimizer = OPTIMIZERS[kwargs.get('optim', 'Adam')](
            self.model.parameters(),
            lr=kwargs.get('lr_', 1e-3),
            betas=kwargs.get('betas_', (0.9, 0.999)),
            eps=kwargs.get('eps-', 1e-8),
            weight_decay=kwargs.get('wd_', 0),
        )

    def new_epoch(self):
        self.epoch += 1
        self.iteration = 0

    def update(self, ):
        self.optimizer.zero_grad()

        batch = self.data_loader.__iter__().__next__()
        loss = self.calc_loss(batch)
        loss.backward()

        self.optimizer.step()
        self.iteration += 1
        if len(self.data_loader) == self.iteration:
            self.new_epoch()

    def calc_loss(self, batch):

        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        out = self.model(images)
        loss = self.loss_fn(out, labels)
        report_mod.report({'loss': round(loss.item(), 5)}, self.model)

        if self.compute_accuracy:
            _, preds = torch.max(out, 1)
            corrects = torch.sum(preds == labels)
            report_mod.report({'accuracy': round((corrects.double() / len(preds)).item(), 5)}, self.model)

        return loss

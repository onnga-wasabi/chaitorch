import torch

import chaitorch.utils.reporter as report_mod

OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
}


class Updater(object):

    loss_fn = torch.nn.CrossEntropyLoss()

    def __init__(self, model, data_loader, device='cpu', compute_accuracy=False, **kwargs):

        self.model = model.to(device)
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
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
            eps=kwargs.get('eps_', 1e-8),
            weight_decay=kwargs.get('wd_', 0),
        )

    def new_epoch(self):
        self.epoch += 1
        self.iteration = 0
        del(self.data_iter)
        self.data_iter = iter(self.data_loader)

    def update(self):
        self.optimizer.zero_grad()

        batch = next(self.data_iter)
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


class TripletLossUpdater(Updater):

    loss_fn = torch.nn.modules.loss.TripletMarginLoss()

    def calc_loss(self, batch):
        x_as, x_ps, x_ns = batch
        x_as = x_as.to(self.device)
        x_ps = x_ps.to(self.device)
        x_ns = x_ns.to(self.device)
        a_out = self.model(x_as)
        p_out = self.model(x_ps)
        n_out = self.model(x_ns)
        loss = self.loss_fn(a_out, p_out, n_out)
        report_mod.report({'loss': round(loss.item(), 5)}, self.model)
        return loss

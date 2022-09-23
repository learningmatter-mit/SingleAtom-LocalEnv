import shutil
import sys
import time
from typing import Dict

import numpy as np
import torch
from persite_painn.utils.cuda import batch_to
from torch.optim.lr_scheduler import ReduceLROnPlateau

MAX_EPOCHS = 100
BEST_METRIC = 1e10
BEST_LOSS = 1e10
EARLY_STOP_VAL = 100
EARLY_STOP_TRAIN = 0.001


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    def __init__(
        self,
        model_path,
        model,
        output_key,
        loss_fn,
        metric_fn,
        optimizer,
        scheduler,
        train_loader,
        validation_loader,
        normalizer=None,
    ):
        self.model_path = model_path
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.output_key = output_key
        self.normalizer = normalizer

    def train(
        self,
        device,
        start_epoch=0,
        n_epochs=MAX_EPOCHS,
        best_loss=BEST_LOSS,
        best_metric=BEST_METRIC,
        early_stop=[EARLY_STOP_VAL, EARLY_STOP_TRAIN],
        save_results=True,
    ):

        # switch to train mode
        self.to(device)
        self.model.train()
        # train model
        train_losses = []
        train_metrics = []
        val_losses = []
        val_metrics = []
        count = 0
        for epoch in range(start_epoch, n_epochs):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            metrics = AverageMeter()
            end = time.time()
            for i, batch in enumerate(self.train_loader):
                self.model.zero_grad(set_to_none=True)
                batch = batch_to(batch, device)
                # measure data loading time
                data_time.update(time.time() - end)
                target = batch[self.output_key]

                if self.normalizer is not None:
                    target = self.normalizer[self.output_key].norm(target)
                output = self.model(batch)[self.output_key]

                if self.model.multifidelity:
                    target_fidelity = batch["fidelity"]
                    target_fidelity = self.normalizer["fidelity"].norm(target_fidelity)
                    output_fidelity = self.model(batch)["fidelity"]

                    loss = self.loss_fn(output, target) + self.loss_fn(
                        output_fidelity, target_fidelity
                    )
                    metric = self.metric_fn(output, target) + self.metric_fn(
                        output_fidelity, target_fidelity
                    )
                else:
                    loss = self.loss_fn(output, target)
                    metric = self.metric_fn(output, target)

                # measure accuracy and record loss
                losses.update(loss.data.cpu().item(), target.size(0))
                metrics.update(metric.cpu().item(), target.size(0))

                # compute gradient and do optim step
                loss.backward()
                # Gradient clipping maybe helpful for spectra learning
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.2)
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Metric {metrics.val:.3f}  ({metrics.avg:.3f})".format(
                            epoch,
                            i,
                            len(self.train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            metrics=metrics,
                        )
                    )
            train_losses.append(losses.avg)
            train_metrics.append(metrics.avg)
            val_loss, val_metric = self.validate(device=device)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)

            if val_loss != val_loss:
                print("Exit due to NaN")
                sys.exit(1)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            best_metric = min(val_metric, best_metric)
            if self.normalizer is not None:
                normalizer_dict = {}
                for key, val in self.normalizer:
                    normalizer_dict[key] = val.state_dict()
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_metric": best_metric,
                        "best_loss": best_loss,
                        "optimizer": self.optimizer.state_dict(),
                        "normalizer": normalizer_dict,
                    },
                    is_best,
                    self.model_path,
                )
            else:
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_metric": best_metric,
                        "best_loss": best_loss,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best,
                    self.model_path,
                )
            # Evaluate when to end training on account of no MAE improvement
            if is_best:
                count = 0
            else:
                count += 1
            if count > early_stop[0] and losses.avg < early_stop[1]:
                break

        if save_results:
            with open(f"{self.model_path}/results.log", "w") as f:
                f.write("| Epoch | Train_L | Train_M | Vali_L | Vali_M |\n")
                for i in range(len(train_losses)):
                    f.write(
                        f"{i}  {train_losses[i]:.4f}  {train_metrics[i]:.4f}  {val_losses[i]:.4f}  {val_metrics[i]:.4f}\n"
                    )

    def validate(self, device, test=False):
        """Validate the current state of the model using the validation set"""
        self.to(device=device)
        self.model.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        metrics = AverageMeter()

        if test:
            test_targets = []
            test_preds = []
            test_ids = []

        end = time.time()

        for val_batch in self.validation_loader:
            val_batch = batch_to(val_batch, device)
            target = val_batch[self.output_key]

            if self.normalizer is not None:
                target = self.normalizer[self.output_key].norm(target)
            output = self.model(val_batch)[self.output_key]

            if self.model.multifidelity:
                target_fidelity = val_batch["fidelity"]
                target_fidelity = self.normalizer["fidelity"].norm(target_fidelity)
                output_fidelity = self.model(val_batch)["fidelity"]

                loss = self.loss_fn(output, target) + self.loss_fn(
                    output_fidelity, target_fidelity
                )
                metric = self.metric_fn(output, target) + self.metric_fn(
                    output_fidelity, target_fidelity
                )
            else:
                loss = self.loss_fn(output, target)
                metric = self.metric_fn(output, target)

            losses.update(loss.data.cpu().item(), target.size(0))
            metrics.update(metric.cpu().item(), target.size(0))

            if test:
                test_pred = output.data.cpu()
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_ids += val_batch["name"]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(
            "*Validatoin: \t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Metric {metrics.avg:.3f}".format(
                batch_time=batch_time,
                loss=losses,
                metrics=metrics,
            )
        )

        if test:
            return test_pred, test_target, test_ids
        else:
            return losses.avg, metrics.avg

    def _load_model_state_dict(self, state_dict):
        if self.torch_parallel:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def save_checkpoint(
        self,
        state: Dict,
        is_best: bool,
        path: str,
        filename: str = "checkpoint.pth.tar",
    ):
        saving_filename = path + "/" + filename
        best_filename = path + "/" + "best_model.pth.tar"
        torch.save(state, saving_filename)
        if is_best:
            shutil.copyfile(saving_filename, best_filename)

    def to(self, device):
        """Changes the device"""
        self.model.device = device
        self.model.to(device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())


# TODO should do on fidelity data also?
def test_model(model, output_key, test_loader, metric_fn, device, normalizer=None):
    """
    test the model performances
    Args:
        model: Model,
        output_key: str,
        test_loader: DataLoader,
        metric_fn: metric function,
        device: "cpu" or "cuda",
    Return:
        Lists of prediction, targets, ids, and metric
    """
    model.to(device)
    model.eval()
    test_targets = []
    test_preds = []
    test_ids = []

    metrics = AverageMeter()
    for batch in test_loader:
        batch = batch_to(batch, device)
        target = batch[model.output_keys[0]]
        if normalizer is not None:
            target = normalizer[output_key].norm(target)
        target = target.to("cpu")
        # Compute output
        output = model(batch, inference=True)[output_key]

        # measure accuracy and record loss
        metric = metric_fn(output, target)

        metrics.update(metric.cpu().item(), target.size(0))

        # Rearrange the outputs
        test_pred = output.data.cpu()
        test_target = target.detach().cpu()
        if test_target.shape[0] == batch["name"].shape[0] and test_target.shape[1] == 1:
            if normalizer is not None:
                test_preds += normalizer[output_key].denorm(test_pred).view(-1).tolist()
                test_targets += (
                    normalizer[output_key].denorm(test_target).view(-1).tolist()
                )
            else:
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()

        if test_target.shape[0] == batch["name"].shape[0] and test_target.shape[1] > 1:
            if normalizer is not None:
                test_preds += normalizer[output_key].denorm(test_pred).tolist()
                test_targets += normalizer[output_key].denorm(test_target).tolist()
            else:
                test_preds += test_pred.tolist()
                test_targets += test_target.view.tolist()

        elif test_target.shape[0] == batch["nxyz"].shape[0]:
            batch_ids = []
            count = 0
            num_bin = []
            for i, val in enumerate(batch["num_atoms"].detach().cpu().numpy()):
                count += val
                num_bin.append(count)
                if i == 0:
                    change = list(np.arange(val))
                else:
                    adding_val = num_bin[i - 1]
                    change = list(np.arange(val) + adding_val)
                batch_ids.append(change)

            if normalizer is not None:
                if test_target.shape[1] == 1:
                    test_preds += [
                        normalizer[output_key].denorm(test_pred[i]).view(-1).tolist()
                        for i in batch_ids
                    ]
                    test_targets += [
                        normalizer[output_key].denorm(test_target[i]).view(-1).tolist()
                        for i in batch_ids
                    ]
                else:
                    test_preds += [
                        normalizer[output_key].denorm(test_pred[i]).tolist()
                        for i in batch_ids
                    ]
                    test_targets += [
                        normalizer[output_key].denorm(test_target[i]).tolist()
                        for i in batch_ids
                    ]

            else:
                if test_target.shape[1] == 1:
                    test_preds += [test_pred[i].view(-1).tolist() for i in batch_ids]
                    test_targets += [
                        test_target[i].view(-1).tolist() for i in batch_ids
                    ]
                else:
                    test_preds += [test_pred[i].tolist() for i in batch_ids]
                    test_targets += [test_target[i].tolist() for i in batch_ids]

        test_ids += batch["name"].detach().tolist()

    return test_preds, test_targets, test_ids, metrics.avg

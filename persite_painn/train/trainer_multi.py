import shutil
import sys
import time
from typing import Dict

import torch
from nff.utils.cuda import batch_to
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
        use_deivce = device
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
                batch = batch_to(batch, use_deivce)
                # measure data loading time
                data_time.update(time.time() - end)
                target = batch["target"]

                if self.normalizer is not None:
                    target = self.normalizer.norm(target)
                    target = target.to(device)
                # print(f"target: {target.size()}")
                output = self.model(batch)[self.output_key]
                output = output.to(device)
                # print(f"output: {output.size()}")
                # output = torch.cat(self.model(batch)["features"])

                loss = self.loss_fn(output, target, torch_device=device).mean()

                # measure accuracy and record loss
                metric = self.metric_fn(output, target, torch_device=device).mean()
                losses.update(loss.data.cpu().item(), target.size(0))
                metrics.update(metric.cpu().item(), target.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
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
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_metric": best_metric,
                        "best_loss": best_loss,
                        "optimizer": self.optimizer.state_dict(),
                        "normalizer": self.normalizer.state_dict(),
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
        self.model.eval()
        use_device = device
        batch_time = AverageMeter()
        losses = AverageMeter()
        metrics = AverageMeter()

        if test:
            test_targets = []
            test_preds = []
            test_ids = []

        end = time.time()
        # val_losses = []
        # val_stmse_errors = []
        for val_batch in self.validation_loader:
            val_batch = batch_to(val_batch, use_device)
            target = val_batch["target"]
            if self.normalizer is not None:
                target = self.normalizer.norm(target)
                target = target.to(device)

            # Compute output
            output = self.model(val_batch)[self.output_key]
            output = output.to(device)

            # output = torch.cat(self.model(batch)["features"])

            loss = self.loss_fn(output, target, torch_device=device).mean()
            # loss = stmse(output, target_var, torch_device=device).mean()
            # measure accuracy and record loss
            metric = self.metric_fn(output, target, torch_device=device).mean()
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

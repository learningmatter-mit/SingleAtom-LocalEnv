import argparse
import json
import os
import pickle as pkl
import sys

import numpy as np
import torch
from nff.data import collate_dicts, split_train_validation_test
from nff.utils.cuda import batch_to
from torch.optim import SGD, Adam, Adadelta, AdamW, NAdam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from persite_painn.data.dataset import PerSiteDataset
from persite_painn.train import Trainer, get_model, Normalizer
from persite_painn.train import (
    stmse_operation,
    mae_operation,
    mse_operation,
    sid_operation,
    sis_operation,
)

parser = argparse.ArgumentParser(
    description="Per-site Crystal Graph Convolutional Neural Networks"
)
parser.add_argument("path_to_data", help="path to directory with data")
parser.add_argument("--site_prop", help="name of site property to predict")
parser.add_argument(
    "--cache",
    default="dataset_cache",
    type=str,
    help="cache where data is / will be stored",
)
parser.add_argument(
    "--modelparams",
    default="modelparams.json",
    type=str,
    help="json file of model parameters",
)
parser.add_argument(
    "--workers", default=0, type=int, help="number of data loading workers"
)
parser.add_argument(
    "--epochs", default=1500, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("-b", "--batch_size", default=64, type=int, help="mini-batch size")
parser.add_argument("--loss_fn", default="SID", type=str, help="choose a loss fn")
parser.add_argument("--metric_fn", default="STMSE", type=str, help="choose a metric fn")
parser.add_argument("--optim", default="Adam", type=str, help="choose an optimizer")
parser.add_argument("--lr", default=0.0007, type=float, help="initial learning rate")
parser.add_argument(
    "--lr_milestones", default=[100], type=int, help="milestones for scheduler"
)
parser.add_argument("--momentum", default=1.2, type=float, help="momentum")
parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
parser.add_argument("--sched", default="multi_step", type=str, help="scheduler")
parser.add_argument(
    "--lr_update_rate", default=30, type=int, help="learning rate update rate"
)
parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint")
parser.add_argument(
    "--val_size",
    default=0.15,
    type=float,
    help="ratio of validation data to be loaded",
)
parser.add_argument(
    "--test_size", default=0.15, type=float, help="ratio of test data to be loaded"
)
parser.add_argument("--savedir", default="./results", type=str, help="saving directory")
parser.add_argument("--cuda", default=3, type=int, help="GPU setting")
parser.add_argument("--device", default="cuda", type=str, help="CPU or cuda")
parser.add_argument(
    "--early_stop_val",
    default=100,
    type=int,
    help="early stopping condition for validation loss update count",
)
parser.add_argument(
    "--early_stop_train",
    default=0.05,
    type=float,
    help="early stopping condition for train loss tolerance",
)
args = parser.parse_args(sys.argv[1:])

assert torch.cuda.is_available(), "cuda is not available"
torch.cuda.set_device(args.cuda)
CUDA_LAUNCH_BLOCKING = 1


def main():

    global args

    # load data
    if os.path.exists(args.cache):
        print("Cached dataset exists...")
        dataset = torch.load(args.cache)
        print(f"Number of Data: {len(dataset)}")
    else:

        data = pkl.load(open(args.path_to_data, "rb"))
        samples = [[id_, struct] for id_, struct in data.items()]
        dataset = PerSiteDataset(
            samples=samples, prop_to_predict=args.site_prop, dataset_cache=args.cache
        )
        print(f"Number of Data: {len(dataset)}")

    train_set, val_set, test_set = split_train_validation_test(
        dataset, val_size=args.val_size, test_size=args.test_size
    )
    # Load model_params
    with open(args.modelparams, "r") as f:
        modelparams = json.load(f)

    model = get_model(modelparams, model_type="PainnAtomwise")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Normalizer
    if not modelparams["spectra"]:
        targs = []
        for batch in train_set:
            targs.append(batch["site_prop"])
        targs = torch.concat(targs, dim=0)
        normalizer = Normalizer(targs)
    else:
        normalizer = None
    model.cuda()

    # Set optimizer
    if args.optim == "SGD":
        print("SGD Optimizer")
        optimizer = SGD(
            trainable_params,
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "Adam":
        print("Adam Optimizer")
        optimizer = Adam(
            trainable_params, args.lr, eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == "Nadam":
        print("NAdam Optimizer")
        optimizer = NAdam(
            trainable_params, args.lr, eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == "AdamW":
        print("AdamW Optimizer")
        optimizer = AdamW(
            trainable_params, args.lr, eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == "Adadelta":
        print("Adadelta Optimizer")
        optimizer = Adadelta(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NameError("Only SGD or Adam or Adadelta is allowed as --optim")
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.start_epoch != 0 and normalizer is not None:
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"]
                normalizer.load_state_dict(checkpoint["normalizer"])
            elif args.start_epoch == 0 and normalizer is not None:
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                normalizer.load_state_dict(checkpoint["normalizer"])
            elif args.start_epoch == 0 and normalizer is None:
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"]
            elif args.start_epoch == 0 and normalizer is None:
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        best_metric = 1e10
        best_loss = 1e10

    # Set loss function
    if args.loss_fn == "SID":
        loss_fn = sid_operation
    elif args.loss_fn == "MSE":
        loss_fn = mse_operation
    else:
        raise NameError("Only SID or MSE is allowed as --loss_fn")
    # Set metric function
    if args.metric_fn == "STMSE":
        metric_fn = stmse_operation
    elif args.metric_fn == "MAE":
        metric_fn = mae_operation
    elif args.metric_fn == "SIS":
        metric_fn = sis_operation
    else:
        raise NameError("Only STMSE or MAE or SIS is allowed as --metric_fn")
    # Set scheduler
    if args.sched == "cos_anneal":
        print("Cosine anneal scheduler")
        scheduler = CosineAnnealingLR(optimizer, args.lr_update_rate)
    elif args.sched == "cos_anneal_warm_restart":
        print("Cosine anneal with warm restarts scheduler")
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.lr_update_rate)
    elif args.sched == "reduce_on_plateau":
        print("Reduce on plateau scheduler")
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.2,
            threshold=0.01,
            verbose=True,
            threshold_mode="abs",
            patience=20,
        )
    elif args.sched == "multi_step":
        print("Multi-step scheduler")
        lr_milestones = np.arange(
            args.lr_update_rate, args.epochs + args.lr_update_rate, args.lr_update_rate
        )
        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    else:
        raise NameError(
            "Choose --sched within cos_anneal, reduce_on_plateau, multi_stp"
        )
    # Set DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=collate_dicts,
        sampler=RandomSampler(train_set),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, collate_fn=collate_dicts
    )
    early_stop = [args.early_stop_val, args.early_stop_train]

    # Temp
    scheduler2 = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.2,
        threshold=0.01,
        verbose=True,
        threshold_mode="abs",
        patience=20,
    )
    # set Trainer
    trainer = Trainer(
        model_path=args.savedir,
        model=model,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=val_loader,
        output_key=args.site_prop,
        normalizer=normalizer,
    )
    # Train
    trainer.train(
        device=args.device,
        scheduler2=scheduler2,
        start_epoch=args.start_epoch,
        n_epochs=args.epochs,
        best_loss=best_loss,
        best_metric=best_metric,
        early_stop=early_stop,
    )

    # Test results
    test_targets = []
    test_preds = []
    test_ids = []
    best_checkpoint = torch.load(f"{args.savedir}/model_best.pth.tar")
    model.load_state_dict(best_checkpoint["state_dict"])

    model.eval()
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, collate_fn=collate_dicts
    )
    for test_batch in test_loader:
        use_device = args.device
        test_batch = batch_to(test_batch, use_device)
        # target
        target = test_batch["site_prop"]

        # Compute output
        output = model(test_batch)[args.site_prop]
        test_target = target.detach().cpu()
        if normalizer is not None:
            test_output = normalizer.denorm(output.detach().cpu())
        else:
            test_output = output.detach().cpu()

        # Batch_ids
        batch_ids = []
        count = 0
        num_bin = []
        for i, val in enumerate(test_batch["num_atoms"].detach().cpu().numpy()):
            count += val
            num_bin.append(count)
            if i == 0:
                change = list(np.arange(val))
            else:
                adding_val = num_bin[i - 1]
                change = list(np.arange(val) + adding_val)
            batch_ids.append(change)

        test_targets += [test_target[i].numpy() for i in batch_ids]
        test_preds += [test_output[i].numpy() for i in batch_ids]
        test_ids += test_batch["name"].detach().tolist()

    # Save Results
    pkl.dump(test_preds, open(f"{args.savedir}/test_preds.pkl", "wb"))
    pkl.dump(test_targets, open(f"{args.savedir}/test_targs.pkl", "wb"))
    pkl.dump(test_ids, open(f"{args.savedir}/test_ids.pkl", "wb"))

    train_ids = []
    for item in train_set:
        train_ids.append(item["name"].item())
    val_ids = []
    for item in val_set:
        val_ids.append(item["name"].item())
    pkl.dump(train_ids, open(f"{args.savedir}/train_ids.pkl", "wb"))
    pkl.dump(val_ids, open(f"{args.savedir}/val_ids.pkl", "wb"))


if __name__ == "__main__":
    main()

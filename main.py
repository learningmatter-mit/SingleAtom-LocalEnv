import argparse
import os
import pickle as pkl
import sys

import torch
from nff.data import collate_dicts

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from persite_painn.data.builder import build_dataset, split_train_validation_test
from persite_painn.nn.builder import load_params, get_model
from persite_painn.train.builder import get_scheduler, get_optimizer
from persite_painn.utils.train_utils import Normalizer

from persite_painn.train.trainer import Trainer, test
from persite_painn.train import (
    stmse_operation,
    mae_operation,
    mse_operation,
    sid_operation,
    sis_operation,
)

parser = argparse.ArgumentParser(description="Per-site PaiNN")
parser.add_argument("--data", default="", type=str, help="path to data")
parser.add_argument("--target", help="name of target to predict")
parser.add_argument("--cache",
                    default="dataset_cache",
                    type=str,
                    help="cache where data is / will be stored")
parser.add_argument("--details",
                    default="details.json",
                    type=str,
                    help="json file of model parameters")
parser.add_argument("--savedir",
                    default="./results",
                    type=str,
                    help="saving directory")
parser.add_argument("--workers",
                    default=0,
                    type=int,
                    help="number of data loading workers")
parser.add_argument("--epochs",
                    default=1500,
                    type=int,
                    help="number of total epochs to run")
parser.add_argument("--start_epoch",
                    default=0,
                    type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument("-b",
                    "--batch_size",
                    default=64,
                    type=int,
                    help="mini-batch size")
parser.add_argument("--loss_fn",
                    default="MSE",
                    type=str,
                    help="choose a loss fn")
parser.add_argument("--metric_fn",
                    default="MAE",
                    type=str,
                    help="choose a metric fn")
parser.add_argument("--optim",
                    default="Adam",
                    type=str,
                    help="choose an optimizer")
parser.add_argument("--lr",
                    default=0.0005,
                    type=float,
                    help="initial learning rate")
parser.add_argument("--weight_decay",
                    default=0.01,
                    type=float,
                    help="weight decay")
parser.add_argument("--print_freq",
                    default=10,
                    type=int,
                    help="print frequency")
parser.add_argument("--sched",
                    default="reduce_on_plateau",
                    type=str,
                    help="scheduler")
parser.add_argument("--resume",
                    default="",
                    type=str,
                    help="path to latest checkpoint")
parser.add_argument("--val_size",
                    default=0.1,
                    type=float,
                    help="ratio of validation data to be loaded")
parser.add_argument("--test_size",
                    default=0.1,
                    type=float,
                    help="ratio of test data to be loaded")
parser.add_argument("--cuda", default=2, type=int, help="GPU setting")
parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
parser.add_argument(
    "--early_stop_val",
    default=50,
    type=int,
    help="early stopping condition for validation loss update count")
parser.add_argument("--early_stop_train",
                    default=0.01,
                    type=float,
                    help="early stopping condition for train loss tolerance")
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="Seed of random initialization to control the experiment")


def main(args):
    # Load details
    details, modelparams, model_type = load_params(args.details)

    # Load data
    if os.path.exists(args.cache):
        print("Cached dataset exists...")
        dataset = torch.load(args.cache)
        print(f"Number of Data: {len(dataset)}")
    else:
        try:
            data = pkl.load(open(args.data, "rb"))
        except ValueError:
            print("Path to data should be given --data")
        else:
            dataset = build_dataset(
                raw_data=data,
                prop_to_predict=args.target,
                cutoff=modelparams["cutoff"],
                site_prediction=details["site_prediction"],
                seed=args.seed,
            )
            print(f"Number of Data: {len(dataset)}")
            print("Done creating dataset, caching...")
            dataset.save(args.cache)
            print("Done caching dataset")

    train_set, val_set, test_set = split_train_validation_test(
        dataset,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed)

    # Get model
    model = get_model(modelparams,
                      model_type=model_type,
                      site_prediction=details["site_prediction"],
                      spectra=details["spectra"],
                      multifidelity=details["multifidelity"])
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Normalizer
    if not details["spectra"]:
        targs = []
        for batch in train_set:
            targs.append(batch["target"])
        targs = torch.concat(targs).view(-1)
        normalizer = Normalizer(targs)
    else:
        normalizer = None
    # model.to(device)
    # Set optimizer
    optimizer = get_optimizer(
        optim=args.optim,
        trainable_params=trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
            elif args.start_epoch == 0 and normalizer is not None:
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                normalizer.load_state_dict(checkpoint["normalizer"])
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
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]))
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
    scheduler = get_scheduler(sched=args.sched,
                              optimizer=optimizer,
                              epochs=args.epochs)

    # Set DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
        sampler=RandomSampler(train_set),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
    )
    early_stop = [args.early_stop_val, args.early_stop_train]

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
        output_key=args.target,
        normalizer=normalizer,
    )
    # Train
    trainer.train(
        device=args.device,
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
    best_checkpoint = torch.load(f"{args.savedir}/best_model.pth.tar")
    model.load_state_dict(best_checkpoint["state_dict"])

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
    )
    test_targets, test_preds, test_ids, _ = test(
        model=model,
        output_key=args.target,
        test_loader=test_loader,
        metric_fn=metric_fn,
        device=args.device,
        normalizer=normalizer,
    )

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
    args = parser.parse_args(sys.argv[1:])
    if args.device == "cuda":
        assert torch.cuda.is_available(), "cuda is not available"
        CUDA_LAUNCH_BLOCKING = 1
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    main(args)

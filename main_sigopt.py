import argparse
import os
import pickle as pkl
import sys
import wandb
import sigopt

import torch
from torch.utils.data import DataLoader

# from torch.utils.data.sampler import RandomSampler
from persite_painn.data.sampler import ImbalancedDatasetSampler

from persite_painn.data import collate_dicts
from persite_painn.data.builder import build_dataset, split_train_validation_test
from persite_painn.nn.builder import get_model, load_params
from persite_painn.train.builder import get_optimizer, get_scheduler, get_loss_metric_fn
from persite_painn.train.trainer import Trainer
from persite_painn.train.evaluate import test_model
from persite_painn.utils.train_utils import Normalizer
from persite_painn.data.preprocess import convert_site_prop
from persite_painn.utils.wandb_utils import save_artifacts
from persite_painn.utils.sigopt_utils import convert_to_sigopt_params

# Running command
#

parser = argparse.ArgumentParser(description="Per-site PaiNN")
parser.add_argument("--data_raw", default="", type=str, help="path to raw data")
parser.add_argument(
    "--cache",
    default="dataset_cache",
    type=str,
    help="cache where data is / will be stored",
)
parser.add_argument(
    "--details", default="details.json", type=str, help="json file of model parameters"
)
parser.add_argument("--savedir", default="./results", type=str, help="saving directory")
parser.add_argument(
    "--workers", default=0, type=int, help="number of data loading workers"
)
parser.add_argument(
    "--epochs", default=150, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("-b", "--batch_size", default=64, type=int, help="mini-batch size")
parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint")
parser.add_argument("--cuda", default=2, type=int, help="GPU setting")
parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
parser.add_argument(
    "--early_stop_val",
    default=12,
    type=int,
    help="early stopping condition for validation loss update count",
)
parser.add_argument(
    "--early_stop_train",
    default=0.1,
    type=float,
    help="early stopping condition for train loss tolerance",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="Seed of random initialization to control the experiment",
)


def main(args):
    # Load details
    wandb_config, details, modelparams, model_type = load_params(args.details)
    details["epochs"] = args.epochs
    sigoptparams = sigopt.params
    new_details, new_params = convert_to_sigopt_params(
        details, modelparams, sigoptparams
    )
    # wandb Sigopt
    wandb_config.update(new_details)
    wandb_config.update(new_params)
    # wandb_config.update(details)
    # wandb_config.update(modelparams)
    wandb.init(
        project=wandb_config["project"], name=wandb_config["name"], config=wandb_config
    )

    # Load data
    if os.path.exists(args.cache):
        print("Cached dataset exists...")
        dataset = torch.load(args.cache)
        print(f"Number of Data: {len(dataset)}")
    else:
        try:
            data = pkl.load(open(args.data_raw, "rb"))
        except ValueError:
            print("Path to data should be given --data")
        else:
            if details["multifidelity"]:
                new_data = convert_site_prop(
                    data,
                    details["output_keys"],
                    details["fidelity_keys"],
                )
                dataset = build_dataset(
                    raw_data=new_data,
                    cutoff=modelparams["cutoff"],
                    multifidelity=details["multifidelity"],
                    seed=args.seed,
                )
            else:
                new_data = convert_site_prop(data, details["output_keys"])
                dataset = build_dataset(
                    raw_data=new_data,
                    cutoff=modelparams["cutoff"],
                    multifidelity=details["multifidelity"],
                    seed=args.seed,
                )

            print(f"Number of Data: {len(dataset)}")
            print("Done creating dataset, caching...")
            dataset.save(args.cache)
            print("Done caching dataset")

    train_set, val_set, test_set = split_train_validation_test(
        dataset,
        val_size=details["val_size"],
        test_size=details["test_size"],
        seed=args.seed,
    )

    # Normalizer
    normalizer = {}
    targs = []
    for batch in train_set:
        targs.append(batch["target"])

    targs = torch.concat(targs).view(-1)
    print(targs.shape)
    valid_index = torch.bitwise_not(torch.isnan(targs))
    filtered_targs = targs[valid_index]
    print(filtered_targs.shape)
    normalizer_target = Normalizer(filtered_targs)
    normalizer["target"] = normalizer_target
    # modelparams.update({"means": {"target": normalizer_target.mean}})
    # modelparams.update({"stddevs": {"target": normalizer_target.std}})

    if details["multifidelity"]:
        fidelity = []
        for batch in train_set:
            fidelity.append(batch["fidelity"])
        fidelity = torch.concat(fidelity).view(-1)
        print(fidelity.shape)
        valid_index = torch.bitwise_not(torch.isnan(fidelity))
        filtered_fidelity = fidelity[valid_index]
        print(filtered_fidelity.shape)
        normalizer_fidelity = Normalizer(filtered_fidelity)
        normalizer["fidelity"] = normalizer_fidelity
        # modelparams.update(
        #     {
        #         "means": {
        #             "target": normalizer_target.mean,
        #             "fidelity": normalizer_fidelity.mean,
        #         }
        #     }
        # )
        # modelparams.update(
        #     {
        #         "stddevs": {
        #             "target": normalizer_target.mean,
        #             "fidelity": normalizer_fidelity.std,
        #         }
        #     }
        # )
    # Sigopt
    sigopt.log_dataset("data_total_multifidelity")
    sigopt.log_model("persitePainnMultifidelity")

    # Get model Sigopt
    model = get_model(
        new_params,
        model_type=model_type,
        multifidelity=details["multifidelity"],
    )
    # model = get_model(
    #     modelparams,
    #     model_type=model_type,
    #     multifidelity=details["multifidelity"],
    # )
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Set optimizer Sigopt
    optimizer = get_optimizer(
        optim=details["optim"],
        trainable_params=trainable_params,
        lr=sigopt.params.lr,
        weight_decay=sigopt.params.weight_decay,
    )
    # optimizer = get_optimizer(
    #     optim=details["optim"],
    #     trainable_params=trainable_params,
    #     lr=details["lr"],
    #     weight_decay=details["weight_decay"],
    # )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.start_epoch != 0:
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"]
                normalizer.load_state_dict(checkpoint["normalizer"])
            elif args.start_epoch == 0:
                checkpoint = torch.load(args.resume)
                best_metric = checkpoint["best_metric"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                normalizer.load_state_dict(checkpoint["normalizer"])

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
    if details["multifidelity"]:
        loss_coeff = modelparams["loss_coeff"]
        correspondence_keys = {"fidelity": "fidelity", "target": "target"}
    else:
        loss_coeff = {"target": 1.0}
        correspondence_keys = {"target": "target"}
    # Set loss function
    loss_fn = get_loss_metric_fn(
        loss_coeff=loss_coeff,
        correspondence_keys=correspondence_keys,
        operation_name=details["loss_fn"],
        normalizer=normalizer,
    )
    # Set metric function
    metric_fn = get_loss_metric_fn(
        loss_coeff=loss_coeff,
        correspondence_keys=correspondence_keys,
        operation_name=details["metric_fn"],
        normalizer=normalizer,
    )

    # Set scheduler Sigopt
    scheduler = get_scheduler(
        sched=details["sched"], optimizer=optimizer, epochs=sigopt.params.epochs
    )
    # scheduler = get_scheduler(
    #     sched=details["sched"], optimizer=optimizer, epochs=args.epochs
    # )

    # Set DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
        sampler=ImbalancedDatasetSampler("classification", train_set.props),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
    )
    # Save ids
    train_ids = []
    for item in train_set:
        train_ids.append(item["name"].item())
    val_ids = []
    for item in val_set:
        val_ids.append(item["name"].item())

    pkl.dump(train_ids, open(f"{args.savedir}/train_ids.pkl", "wb"))
    pkl.dump(val_ids, open(f"{args.savedir}/val_ids.pkl", "wb"))

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
        normalizer=normalizer,
    )
    # Train Sigopt
    best_metric_score = trainer.train(
        device=args.device,
        start_epoch=args.start_epoch,
        n_epochs=sigopt.params.epochs,
        best_loss=best_loss,
        best_metric=best_metric,
        early_stop=early_stop,
    )
    # best_metric_score = trainer.train(
    #     device=args.device,
    #     start_epoch=args.start_epoch,
    #     n_epochs=args.epochs,
    #     best_loss=best_loss,
    #     best_metric=best_metric,
    #     early_stop=early_stop,
    # )
    # Sigopt Log
    sigopt.log_metric(name="best_mae_error", value=best_metric_score)
    # Test results
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dicts,
    )
    test_targets = []
    test_preds = []

    best_checkpoint = torch.load(f"{args.savedir}/best_model.pth.tar")
    model.load_state_dict(best_checkpoint["state_dict"])

    (
        test_preds,
        test_targets,
        test_ids,
        _,
        test_preds_fidelity,
        test_targets_fidelity,
    ) = test_model(
        model=model,
        test_loader=test_loader,
        metric_fn=metric_fn,
        device="cpu",
        normalizer=normalizer,
        multifidelity=details["multifidelity"],
    )

    # Save Test Results
    pkl.dump(test_ids, open(f"{args.savedir}/test_ids.pkl", "wb"))
    pkl.dump(test_preds, open(f"{args.savedir}/test_preds.pkl", "wb"))
    pkl.dump(test_targets, open(f"{args.savedir}/test_targs.pkl", "wb"))
    if details["multifidelity"]:
        pkl.dump(
            test_preds_fidelity, open(f"{args.savedir}/test_preds_fidelity.pkl", "wb")
        )
        pkl.dump(
            test_targets_fidelity, open(f"{args.savedir}/test_targs_fidelity.pkl", "wb")
        )

    # save wandb artifacts
    save_artifacts(args.savedir)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    # TODO CUDA Settings confusing
    if args.device == "cuda":
        # os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        assert torch.cuda.is_available(), "cuda is not available"

    main(args)
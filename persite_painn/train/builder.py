import numpy as np
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    ReduceLROnPlateau,
)
from torch.optim import SGD, Adam, Adadelta, AdamW, NAdam, RAdam


def get_optimizer(
    optim: str,
    trainable_params,
    lr: float,
    weight_decay: float,
):
    if optim == "SGD":
        print("SGD Optimizer")
        optimizer = SGD(
            trainable_params,
            lr=lr,
            momentum=0.0,
            weight_decay=weight_decay,
        )
    elif optim == "Adam":
        print("Adam Optimizer")
        optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "Nadam":
        print("NAdam Optimizer")
        optimizer = NAdam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "AdamW":
        print("AdamW Optimizer")
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "Adadelta":
        print("Adadelta Optimizer")
        optimizer = Adadelta(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim == "Radam":
        print("RAdam Optimizer")
        optimizer = RAdam(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        raise NameError("Optimizer not implemented --optim")

    return optimizer


def get_scheduler(
    sched: str, optimizer, epochs, lr_update_rate=30, lr_milestones=[100]
):
    if sched == "cos_anneal":
        print("Cosine anneal scheduler")
        scheduler = CosineAnnealingLR(optimizer, lr_update_rate)
    elif sched == "cos_anneal_warm_restart":
        print("Cosine anneal with warm restarts scheduler")
        scheduler = CosineAnnealingWarmRestarts(optimizer, lr_update_rate)
    elif sched == "reduce_on_plateau":
        print("Reduce on plateau scheduler")
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.5,
            threshold=0.01,
            verbose=True,
            threshold_mode="abs",
            patience=15,
        )
    elif sched == "multi_step":
        print("Multi-step scheduler")
        lr_milestones = np.arange(
            lr_update_rate, epochs + lr_update_rate, lr_update_rate
        )
        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    else:
        raise NameError("Choose within cos_anneal, reduce_on_plateau, multi_stp")
    return scheduler

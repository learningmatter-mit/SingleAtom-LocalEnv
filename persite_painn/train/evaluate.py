import numpy as np
from persite_painn.utils import inference, batch_to
from persite_painn.train import AverageMeter


def test_model(
    model,
    output_key,
    test_loader,
    metric_fn,
    device,
    normalizer=None,
    multifidelity=False,
):
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
        target = batch[output_key]
        # Compute output
        output = inference(model, batch, output_key, normalizer, device)
        metric_output = model(batch, inference=True)

        # measure accuracy and record loss
        metric = metric_fn(metric_output, batch)

        metrics.update(metric.cpu().item(), target.size(0))

        # Rearrange the outputs
        test_pred = output.data.cpu()
        test_target = target.detach().cpu()
        if test_target.shape[0] == batch["name"].shape[0] and test_target.shape[1] == 1:
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()

        elif (
            test_target.shape[0] == batch["name"].shape[0] and test_target.shape[1] > 1
        ):
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

            if test_target.shape[1] == 1:
                test_preds += [test_pred[i].view(-1).tolist() for i in batch_ids]
                test_targets += [test_target[i].view(-1).tolist() for i in batch_ids]
            else:
                test_preds += [test_pred[i].tolist() for i in batch_ids]
                test_targets += [test_target[i].tolist() for i in batch_ids]

        if multifidelity:
            test_targets_fidelity = []
            test_preds_fidelity = []
            target_fidelity = batch["fidelity"].detach().cpu()
            # Compute output
            output_fidelity = inference(
                model, batch, "fidelity", normalizer, device
            ).data.cpu()
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
            test_preds_fidelity += [output_fidelity[i].tolist() for i in batch_ids]
            test_targets_fidelity += [target_fidelity[i].tolist() for i in batch_ids]

        else:
            test_targets_fidelity = None
            test_preds_fidelity = None

        test_ids += batch["name"].detach().tolist()

    return (
        test_preds,
        test_targets,
        test_ids,
        metrics.avg,
        test_preds_fidelity,
        test_targets_fidelity,
    )

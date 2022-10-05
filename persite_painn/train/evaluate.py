import numpy as np
import torch
from persite_painn.train import AverageMeter
from persite_painn.utils import batch_to, inference


def test_model(
    model,
    output_key,
    test_loader,
    metric_fn,
    device,
    normalizer=None,
    spectra=False,
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
    test_targets_fidelity = []
    test_preds_fidelity = []
    metric_bin = []
    metrics = AverageMeter()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch_to(batch, device)
            target = batch[output_key]
            # Compute output
            output = inference(model, batch, output_key, normalizer, device)
            if device == "cpu":
                metric_output = model(batch, inference=True)
            else:
                metric_output = model(batch)
            # measure accuracy and record loss
            metric = metric_fn(metric_output, batch)

            if spectra:
                metrics.update(torch.mean(metric).cpu().item(), target.size(0))
            else:
                metrics.update(metric.cpu().item(), target.size(0))

            # Rearrange the outputs
            test_pred = output.data.cpu()
            test_target = target.detach().cpu()
            if (
                test_target.shape[0] == batch["name"].shape[0]
                and test_target.shape[1] == 1
            ):
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()

            elif (
                test_target.shape[0] == batch["name"].shape[0]
                and test_target.shape[1] > 1
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

                if spectra:
                    test_preds += [test_pred[i].tolist() for i in batch_ids]
                    test_targets += [test_target[i].tolist() for i in batch_ids]
                    metric_bin += [metric[i].tolist() for i in batch_ids]
                else:
                    test_preds += [test_pred[i].tolist() for i in batch_ids]
                    test_targets += [test_target[i].tolist() for i in batch_ids]

            if multifidelity:
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
                test_targets_fidelity += [
                    target_fidelity[i].tolist() for i in batch_ids
                ]
            if spectra:
                metric_out = metric_bin
            else:
                metric_out = metrics.avg
            test_ids += batch["name"].detach().tolist()

    return (
        test_preds,
        test_targets,
        test_ids,
        metric_out,
        test_preds_fidelity,
        test_targets_fidelity,
    )

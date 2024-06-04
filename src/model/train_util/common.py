import torch

def calculate_threshold(model):
    all_gradients = torch.cat([p.grad.view(-1)
                              for p in model.parameters() if p.grad is not None])
    gradient_mean = torch.mean(all_gradients)
    gradient_std = torch.std(all_gradients)
    threshold = gradient_mean + gradient_std * 1.96
    return threshold


def clip_gradients(model, threshold=None, use_outliers=False):
    if threshold is None:
        threshold = calculate_threshold(model)
    if use_outliers:
        outliers = [p for p in model.parameters()
                    if p.grad is not None and torch.max(torch.abs(p.grad)) > threshold]
        if outliers:
            max_outlier_value = max(
                [torch.max(torch.abs(p.grad)) for p in outliers])
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                            max_outlier_value)
    else:
        torch.nn.utils.clip_grad_value_(model.parameters(), threshold)

import torch
import numpy as np


# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)
    # Backpropagation
    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    return loss_b.item(), metric_b


def train_loop(dataloader, model, loss_fn, optimizer, device="cpu", print_batch=False):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    loss, metric = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X).to(device)
        y = y.to(device)
        batch_loss, batch_metric = loss_batch(loss_fn, pred, y, optimizer)
        loss += batch_loss
        if batch_metric is not None:
            metric += batch_metric
        if print_batch & (batch % 100 == 0):
            loss, current = batch_loss, (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    loss /= size
    metric /= size
    return loss, metric


def test_loop(dataloader, model, loss_fn, device="cpu"):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, metric = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).to(device)
            y = y.to(device)
            batch_loss, batch_metric = loss_batch(loss_fn, pred, y)
            loss += batch_loss
            if batch_metric is not None:
                metric += batch_metric
    loss /= size
    metric /= size
    return loss, metric

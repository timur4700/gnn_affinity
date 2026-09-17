import torch
from torch import Module, Tensor
from torch_geometric.data import Data


def standard_loss_and_backward(
        batch: Data,
        model: Module,
        loss_func: Module,
) -> Tensor:

    y_true = batch.y

    y_hat = model(batch)
    loss = loss_func(
        y_hat.squeeze(-1), 
        y_true
    )

    loss.backward()

    return loss


def FLAG_loss_and_backward(
        model,
        batch: Data,
        criterion,
        step_size: float=1e-3,
        m: int=3,
        on_hidden_dim: bool=True
):
    n = batch.x.size(0)

    h = (model.hidden_dim if on_hidden_dim else 
         batch.x.size(1))

    delta = torch.empty(
        n, h, 
        device=batch.x.device
    ).uniform_(-step_size, step_size)

    delta.requires_grad_()

    losses = 0

    for _ in range(m):

        pred = model(batch, perturb=delta)
        loss = criterion(
            pred.squeeze(-1), 
            batch.y
        ) / m

        loss.backward()
        losses += loss.detach()

        delta = delta.detach() + step_size * delta.grad.sign()
        delta.requires_grad_()

    return losses
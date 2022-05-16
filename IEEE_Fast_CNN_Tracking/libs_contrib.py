import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import List

def nadam(params: List[Tensor], # Next PyTorch Release
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          mu_products: List[float],
          state_steps: List[int],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          momentum_decay: float,
          eps: float):
    r"""Functional API that performs NAdam algorithm computation.
    See :class:`~torch.optim.NAdam` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu_product = mu_products[i]
        step = state_steps[i]

        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # calculate the momentum cache \mu^{t} and \mu^{t+1}
        mu = beta1 * (1. - 0.5 * (0.96 ** (step * momentum_decay)))
        mu_next = beta1 * (1. - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))
        mu_product = mu_product * mu
        mu_product_next = mu_product * mu * mu_next

        # decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)
        param.addcdiv_(grad, denom, value=-lr * (1. - mu) / (1. - mu_product))
        param.addcdiv_(exp_avg, denom, value=-lr * mu_next / (1. - mu_product_next))

class NAdam(torch.optim.Optimizer):
    r"""Implements NAdam algorithm.
    It has been proposed in `Incorporating Nesterov Momentum into Adam`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum_decay (float, optional): momentum momentum_decay (default: 4e-3)
    .. _Incorporating Nesterov Momentum into Adam:
        https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
    """
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, momentum_decay=4e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= momentum_decay:
            raise ValueError("Invalid momentum_decay value: {}".format(momentum_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, momentum_decay=momentum_decay)
        super(NAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            mu_products = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('NAdam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['mu_product'] = 1.
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    mu_products.append(state['mu_product'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            nadam(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    mu_products,
                    state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    momentum_decay=group['momentum_decay'],
                    eps=group['eps'])

            # update mu_product
            for p, mu_product in zip(params_with_grad, mu_products):
                state = self.state[p]
                state['mu_product'] = state['mu_product'] * beta1 * \
                    (1. - 0.5 * (0.96 ** (state['step'] * group['momentum_decay'])))

        return loss

class EarlyStopping: # https://github.com/Bjarten/early-stopping-pytorch/
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

import torch
from torch.optim.optimizer import Optimizer, required

import math


class AGD(Optimizer):
    r"""Testing implementation of our optimisation method.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        f_star (float): the global optimial minimum value
        lamb (float): the lambda parameter in range (0,1]
        gamma (float, optional): a small user chosen value added to the denominator (default: 0)
    .. note::
        
    """

    def __init__(self, params, f_star, lamb=required, gamma=0, 
                lamb_anneal=False, gamma_anneal=False, eta=0.1):
        if lamb is not required and lamb <= 0.0:
            raise ValueError("Invalid lambda value: {}".format(lamb))
        if gamma < 0.0:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        defaults = dict(f_star=f_star, lamb=lamb, gamma=gamma, 
                        lamb_anneal=lamb_anneal, gamma_anneal=gamma_anneal,
                        eta=eta)
        super(AGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, f_x, epoch=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['lamb_anneal'] and epoch is None:
                raise ValueError("Need to provide epoch for annealing.")
            if group['gamma_anneal'] and epoch is None:
                raise ValueError("Need to provide epoch for annealing.")


            eta = group['eta']

            gTg = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                gTg += torch.sum(p.grad**2)
                
            delta = lamb * (f_x-group['f_star']) / (group['gamma']+gTg)
            bounded = min(delta, eta)
            delta = -bounded


            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                
                p.add_(g, alpha=delta)

        return loss,delta

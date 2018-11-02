import torch
from .optimizer import Optimizer, required


class PLSSGD(Optimizer):
    """ Implements Predictive Local Smoothness for Stochastic Gradient Methods.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        L (float) : upper bound of the smoothness (default: 3)
        eps1 (float) : euqation 10 (default: 0.05)
        eps2 (float) : euqation 10 (default: 0.1)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.

     .. _Predictive Local Smoothness for Stochastic Gradient Methods:
         https://arxiv.org/abs/1805.09386
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, L_upper=3, eps1=0.05, eps2=0.1, element_wise=False, lipschitz_type=0):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if L_upper < 0.0:
            raise ValueError("Invalid L value: {}".format(L_upper))
        if eps1 < 0.0:
            raise ValueError("Invalid eps1 value: {}".format(eps1))
        if eps2 < 0.0:
            raise ValueError("Invalid eps2 value: {}".format(eps2))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, L_upper=L_upper, eps1=eps1, eps2=eps2, element_wise=element_wise, lipschitz_type=lipschitz_type)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PLSSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PLSSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            L_upper = group['L_upper']
            eps1 = group['eps1']
            eps2 = group['eps2']
            element_wise = group['element_wise']
            lipschitz_type = group['lipschitz_type']
            lr = group['lr']


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
               
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    grad_previous = state['grad_previous'] = torch.zeros_like(p.data)
                    weight_previous = state['weight_previous'] = torch.zeros_like(p.data)
                    grad_previous.add_(d_p)                    
                    weight_previous.add_(p.data)                    
                # else:
                #     ## start from the second step
                #     print ("Not implemented")
                #     grad_previous = state['grad_previous'] 
                #     weight_previous = state['weight_previous']

                grad_previous = state['grad_previous']
                weight_previous = state['grad_previous']

                state['step'] += 1
                # tmp = torch.zeros_like(p.data)
                if state['step'] > 1:
                    if element_wise:
                        ## 
                        raise Exception("NOT Implemeted")
                    else: 
                        ## variable 
                        gt_diff = grad_previous - d_p
                        gt_diff_norm = torch.norm(gt_diff)

                        if lipschitz_type == 0:
                            ## w_t - w_{t-1}
                            devided = weight_previous - p.data 
                        elif lipschitz_type == 1:
                            ## g_{t_1}
                            devided = grad_previous
                        elif lipschitz_type == 2:
                            ## g_t
                            devided = p.data
                        else:
                            ## should not happen
                            raise ValueError("lipschitz_type accepts inputs {0, 1, 2}")
                               
                        devided_norm = torch.norm(devided)

                        L = gt_diff_norm / (devided_norm + eps1)
                        # print (gt_diff_norm) 
                        # print (devided_norm) 
                        # print (L)
                        
                        assert (L>=0) , "Oh no! This assertion failed! {}".format(L)
                        L = L.clamp_(0, L_upper)

                ## update stored parameters
                grad_previous.mul_(0.0).add_(1.0, p.grad.data)
                weight_previous.mul_(0.0).add_(1.0, p.data)
 
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if state['step'] == 1:
                    lr_ = lr
                else:
                    lr_ = lr / (L + eps2)
                # print (lr)
                # exit()
                # p.data.add_(-group['lr'], d_p)
                p.data.add_(-lr_, d_p)

        return loss

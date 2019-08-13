import torch

class GradClipper():

    def __init__(self, optimizer, grad_clip=2.):
        self.optimizer = optimizer
        self.grad_clip = grad_clip

        if hasattr(torch.nn.utils, 'clip_grad_norm_'):
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        else:
            self.clip_grad_norm =  torch.nn.utils.clip_grad.clip_grad_norm

    def step(self):
        params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                params.append(p)
        self.clip_grad_norm(params, self.grad_clip)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
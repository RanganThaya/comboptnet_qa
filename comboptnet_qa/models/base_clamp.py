import torch


class PosClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0.01, max=1)  # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class HalfClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0.01, max=0.3)  # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


import torch


class NegClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=-1, max=0)  # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class QuestionClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=2, max=5)  # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

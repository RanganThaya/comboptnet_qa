import collections
import logging
import os
import sys
from ast import arg
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import gurobipy as grb
import numpy as np
import ray
import torch
import torch.nn.functional as F
from gurobipy import GRB, quicksum
from torch import autograd, nn
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

STATUS_MSG = collections.defaultdict(
    lambda: "unknown",
    {
        1: "loaded",
        2: "optimal",
        3: "infeasible",
        4: "inf_or_unbd",
        5: "unbounded",
        6: "cutoff",
        7: "iteration_limit",
        8: "node_limit",
        9: "time_limit",
        10: "solution_limit",
        11: "interrupted",
        12: "numeric",
        13: "suboptimal",
        14: "inprogress",
        15: "user_obj_limit",
    },
)


class DiffSolve(autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, obj, *args):
        # *args, h = args
        args = [arg.detach() for arg in args]
        # if h is None:
        y, status = obj.solve(*args)
        # else:
        # y, status = obj.solve(*args, h)
        ctx.save_for_backward(*args, y.detach(), status.detach())
        ctx.obj = obj
        return y, status

    @staticmethod
    @torch.no_grad()
    def make_basis(y, dy):
        device = y.device
        batch_size, num_vars = y.shape
        val, idx = torch.sort(torch.abs(dy), dim=-1, descending=True)

        tril = torch.tril(torch.ones(num_vars, num_vars, device=device)).expand(
            batch_size, -1, -1
        )
        basis = torch.empty(batch_size, num_vars, num_vars, device=device)
        basis.scatter_(-1, idx[:, None, :].expand(-1, num_vars, -1), tril)
        basis *= torch.sign(dy)[:, None, :]

        pad = torch.cat([val, torch.zeros(batch_size, 1, device=device)], dim=-1)
        coeff = pad[:, :-1] - pad[:, 1:]

        return basis, coeff

    @staticmethod
    @torch.enable_grad()
    def backward(ctx, dy, dstatus):
        a, b, c, y, status = ctx.saved_tensors
        num_constrs, num_vars = a.shape[-2:]

        basis, coeff = DiffSolve.make_basis(y, dy)
        neg = y[:, None, :]
        pos = neg - basis

        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True

        loss = ctx.obj.criterion(a, b, c, pos, neg, coeff)
        loss.backward()

        return None, a.grad, b.grad, c.grad, None


class ILPLoss(nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
    ):
        super().__init__()

        self.tau = tau

        self.hparams = {
            "tau": tau,
        }

    def min(self, x, dim):
        return -self.tau * torch.logsumexp(-x / self.tau, dim=dim)

    def forward(self, a, b, c, pos, neg, coeff=1):
        norm_a = torch.linalg.vector_norm(a, dim=-1)
        a = a / norm_a[:, :, None]
        b = b / norm_a
        a_t = a.transpose(-1, -2)
        b_t = b[:, None, :]

        norm_c = torch.linalg.vector_norm(c, dim=-1)
        c = c / norm_c[:, None]
        c_t = c[:, :, None]

        dist_pos = pos @ a_t + b_t
        dist_neg = neg @ a_t + b_t
        dist_obj = (pos - neg) @ c_t

        raw_loss_pos = torch.sum(F.relu(-dist_pos), dim=-1)
        raw_loss_neg = self.min(F.relu(dist_neg), dim=-1)
        raw_loss_obj = F.relu(dist_obj)[:, :, 0]

        raw_loss = raw_loss_pos + (raw_loss_neg + raw_loss_obj) * (raw_loss_pos == 0)

        return torch.mean(torch.sum(coeff * raw_loss, dim=-1))


@ray.remote
def aux(a, b, c):
    num_constrs, num_vars = a.shape
    sys.stdout = open(os.devnull, "w")
    with grb.Env() as env:
        env.setParam("OutputFlag", 0)
        with grb.Model(env=env) as model:
            model.setParam("Threads", 1)
            model.setParam("OutputFlag", 0)
            variables = [
                model.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name="v" + str(i))
                for i in range(num_vars)
            ]

            model.setObjective(
                quicksum(c * var for c, var in zip(c, variables)),
                GRB.MINIMIZE,
            )
            for a, _b in zip(a, b):
                # for c, var in zip(a, variables):
                # print(c, var, _b * -1)
                model.addConstr(
                    quicksum(c * var for c, var in zip(a, variables)) + _b <= 0
                )
            model.optimize()
            try:
                y = torch.Tensor([v.x for v in model.getVars()])
                # y[i] = torch.Tensor([v.x for v in model.getVars()])
            except grb.GurobiError:
                y = torch.ones(num_vars)
            status = model.status
            sys.stdout = sys.__stdout__
            return y, status


class CombOptNet(nn.Module):
    def __init__(
        self,
        # vtype: str = GRB.BINARY,
        vtype: str = GRB.INTEGER,
        num_workers: int = 1,
        show_tqdm: bool = False,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.vtype = vtype
        # self.env = grb.Env() if env is None else env
        self.exe = ThreadPoolExecutor(num_workers)
        self.show_tqdm = show_tqdm
        if criterion is None:
            criterion = ILPLoss()
        self.criterion = criterion

        self.hparams = {
            "criterion": criterion.hparams,
        }

    def solve(self, *args):
        a = args[0]
        batch_size, _, num_vars = a.shape
        y = torch.empty(batch_size, num_vars, device=a.device)
        status = torch.empty(batch_size, device=a.device, dtype=torch.long)

        args = [arg.cpu().numpy() for arg in args]
        results = ray.get(
            [aux.remote(args[0][i], args[1][i], args[2][i]) for i in range(batch_size)]
        )
        for index, (y_hat, status_hat) in enumerate(results):
            y[index] = y_hat
            status[index] = status_hat

        # list(
        #     tqdm(
        #         self.exe.map(
        #             aux,
        #             np.arange(batch_size),
        #             *[arg.cpu().numpy() for arg in args],
        #         ),
        #         "instances",
        #         total=batch_size,
        #         disable=not self.show_tqdm,
        #     )
        # )

        return y, status

    def forward(self, *args):
        return DiffSolve.apply(self, *args)

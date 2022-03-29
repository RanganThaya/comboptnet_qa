import os
import sys
import warnings
from operator import mod

import gurobipy as gp
import jax.numpy as jnp
import numpy as np
import torch
from gurobipy import GRB, quicksum
from jax import grad

from .comboptnet_utils import (
    check_point_feasibility,
    compute_delta_y,
    signed_euclidean_distance_constraint_point,
    softmin,
    tensor_to_jax,
)
from .utils import ParallelProcessing


class BlackBoxILP(torch.nn.Module):
    def __init__(
        self,
        variable_range,
        num_nodes,
        lambda_val=5e-1,
        tau=None,
        clip_gradients_to_box=True,
        use_canonical_basis=False,
    ):
        super().__init__()
        """
        @param variable_range: dict(lb, ub), range of variables in the ILP
        @param tau: a float/np.float32/torch.float32, the value of tau for computing the constraint gradient
        @param clip_gradients_to_box: boolean flag, if true the gradients are projected into the feasible hypercube
        @param use_canonical_basis: boolean flag, if true the canonical basis is used instead of delta basis
        """
        self.solver_params = dict(
            lambda_val=lambda_val,
            tau=tau,
            variable_range=variable_range,
            clip_gradients_to_box=clip_gradients_to_box,
            use_canonical_basis=use_canonical_basis,
            parallel_processing=ParallelProcessing(),
            num_nodes=num_nodes,
        )
        self.solver = DifferentiableILPsolver

    def forward(self, cost_vector, constraints):
        """
        Forward pass of CombOptNet running a differentiable ILP solver
        @param cost_vector: torch.Tensor of shape (bs, num_variables) with batch of ILP cost vectors
        @param constraints: torch.Tensor of shape (bs, num_const, num_variables + 1) or (num_const, num_variables + 1)
                            with (potentially batch of) ILP constraints
        @return: torch.Tensor of shape (bs, num_variables) with integer values capturing the solution of the ILP
        """
        if len(constraints.shape) == 2:
            bs = cost_vector.shape[0]
            constraints = torch.stack(bs * [constraints])
        y = self.solver.apply(cost_vector, constraints, self.solver_params)
        return y


class DifferentiableILPsolver(torch.autograd.Function):
    """
    Differentiable ILP solver as a torch.Function
    """

    @staticmethod
    def forward(ctx, cost_vector, constraints, params):
        """
        Implementation of the forward pass of a batched (potentially parallelized) ILP solver.
        @param ctx: context for backpropagation
        @param cost_vector: torch.Tensor of shape (bs, num_variables) with batch of ILp cost vectors
        @param constraints: torch.Tensor of shape (bs, num_const, num_variables + 1) with batch of  ILP constraints
        @param params: a dict of additional params. Must contain:
                tau: a float/np.float32/torch.float32, the value of tau for computing the constraint gradient
                clip_gradients_to_box: boolean flag, if true the gradients are projected into the feasible hypercube
        @return: torch.Tensor of shape (bs, num_variables) with integer values capturing the solution of the ILP,
                 torch.Tensor of shape (bs) with 0/1 values, where 1 corresponds to an infeasible ILP instance
        """
        device = constraints.device
        maybe_parallelize = params["parallel_processing"].maybe_parallelize

        dynamic_args = [
            {
                "cost_vector": cost_vector,
                "constraints": const,
                "num_nodes": params["num_nodes"],
            }
            for cost_vector, const in zip(
                cost_vector.cpu().detach().numpy(), constraints.cpu().detach().numpy()
            )
        ]

        result = maybe_parallelize(ilp_solver, params["variable_range"], dynamic_args)
        y, infeasibility_indicator = [
            torch.from_numpy(np.array(res)).to(device) for res in zip(*result)
        ]

        ctx.params = params
        ctx.save_for_backward(cost_vector, constraints, y, infeasibility_indicator)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        """
        Backward pass computation.
        @param ctx: context from the forward pass
        @param y_grad: torch.Tensor of shape (bs, num_variables) describing the incoming gradient dy for y
        @return: torch.Tensor of shape (bs, num_variables) gradient dL / cost_vector
                 torch.Tensor of shape (bs, num_constraints, num_variables + 1) gradient dL / constraints
        """
        cost_vector, constraints, y_old, infeasibility_indicator = ctx.saved_tensors
        grad_output_numpy = y_grad.detach().cpu().numpy()

        # print(grad_output_numpy.nonzero())
        # print(grad_output_numpy[0].reshape(24, 24))
        cost_vector = cost_vector.cpu().detach().numpy()
        cost_vector = cost_vector + ctx.params["lambda_val"] * grad_output_numpy
        dynamic_args = [
            {
                "cost_vector": cost_vector,
                "constraints": const,
                "num_nodes": ctx.params["num_nodes"],
            }
            for cost_vector, const, y_old_val in zip(
                cost_vector,
                constraints.cpu().detach().numpy(),
                y_old.cpu().detach().numpy(),
            )
        ]
        maybe_parallelize = ctx.params["parallel_processing"].maybe_parallelize
        # print(dynamic_args)
        result = maybe_parallelize(
            ilp_solver,
            ctx.params["variable_range"],
            dynamic_args,
        )
        device = constraints.device
        y_new, infeasibility_indicator = [
            torch.from_numpy(np.array(res)).to(device) for res in zip(*result)
        ]
        cost_vector_grad = -(y_old - y_new) / ctx.params["lambda_val"]

        return cost_vector_grad, None, None


def ilp_solver(cost_vector, constraints, lb, ub, num_nodes, start=None):
    """
    ILP solver using Gurobi. Computes the solution of a single integer linear program
    y* = argmin_y (c * y) subject to A @ y + b <= 0, y integer, lb <= y <= ub

    @param cost_vector: np.array of shape (num_variables) with cost vector of the ILP
    @param constraints: np.array of shape (num_const, num_variables + 1) with constraints of the ILP
    @param lb: float, lower bound of variables
    @param ub: float, upper bound of variables
    @return: np.array of shape (num_variables) with integer values capturing the solution of the ILP,
             boolean flag, where true corresponds to an infeasible ILP instance
    """
    A, b = constraints[:, :-1], constraints[:, -1]
    num_constraints, num_variables = A.shape
    sys.stdout = open(os.devnull, "w")
    with gp.Env() as env:
        env.setParam("OutputFlag", 0)
        sys.stdout = open(os.devnull, "w")
        with gp.Model(env=env) as model:
            model.setParam("OutputFlag", 0)
            model.setParam("Threads", 1)
            # model.setParam("Method", 3)
            # model.setParam("MIPFocus", 1)

            # variables = [
            #     model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER, name="v" + str(i))
            #     for i in range(num_variables)
            # ]
            variables = model.addMVar(num_variables, lb=lb, ub=ub, vtype=GRB.INTEGER)
            # if start is not None:
            #     for i, var in enumerate(model.getVars()):
            #         var.start = start[i]
            # model.update()

            # print()
            identity = 1 - np.identity(num_nodes)
            identity = identity.reshape((num_nodes * num_nodes))
            cost_vector = identity * cost_vector
            # print(identity)

            model.setObjective(
                quicksum(c * var for c, var in zip(cost_vector, variables)),
                GRB.MINIMIZE,
            )

            # model.setObjective(quicksum(cost_vector * variables, GRB.MINIMIZE))

            # for a, _b in zip(A, b):
            #     # for c, var in zip(a, variables):
            #     # print(c, var, _b * -1)
            #     model.addConstr(
            #         quicksum(c * var for c, var in zip(a, variables)) + _b <= 0
            #     )
            model.addConstr(A @ variables + b <= 0)
            model.optimize()
            try:
                y = np.array([v.x for v in model.getVars()])
                infeasible = False
            except AttributeError:
                warnings.warn(
                    f"Infeasible ILP encountered. Dummy solution should be handled as special case."
                )
                y = np.zeros_like(cost_vector)
                infeasible = True
            sys.stdout = sys.__stdout__
            return y, infeasible

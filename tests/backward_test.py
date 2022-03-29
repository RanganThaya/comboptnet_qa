import unittest

import cvxpy as cp
import numpy as np
import torch
from comboptnet_qa.models.comboptnet.comboptnet import CombOptNetModule
from cvxpy.expressions.cvxtypes import problem
from cvxpylayers.torch import CvxpyLayer
from dynaconf import settings
from jax._src.dtypes import result_type

NO_OF_NODES = 9

node_relevance = np.array([0, 0.2, 0.3, 0.4, 0.1, 0.2, 0.6, 0.7, 0.9])
groudning_nodes = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0])
abstract_nodes = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
outgoing_edges = np.array(
    [
        [0, 1, 1, 1, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

inter_edges = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


def build_constraints(
    num_nodes,
    is_abstract_fact,
    abstract_limit,
    inter_edges_weights,
    outgoing_edges_weigths,
):

    NO_OF_NODES = num_nodes
    num_variables = NO_OF_NODES + NO_OF_NODES * NO_OF_NODES

    incoming_edges = np.where(
        outgoing_edges_weigths.T != 0,
        np.ones_like(outgoing_edges_weigths),
        np.zeros_like(outgoing_edges_weigths),
    )
    inter_edges = np.where(
        inter_edges_weights != 0,
        np.ones_like(inter_edges_weights),
        np.zeros_like(inter_edges_weights),
    )

    # root constraint
    root_constraint_A = torch.zeros((1, num_variables))
    root_constraint_A[0, 0] = -1
    root_constraint_b = torch.Tensor([[1]])

    # incoming constraints
    incoming_constraints_A = torch.Tensor(np.identity(NO_OF_NODES) - incoming_edges)
    incoming_constraints_A = torch.cat(
        (
            incoming_constraints_A,
            torch.zeros((NO_OF_NODES, NO_OF_NODES * NO_OF_NODES)),
        ),
        dim=1,
    )
    incoming_constraints_b = torch.zeros((NO_OF_NODES, 1))

    # edge selection constraint
    edge_selection_A = torch.zeros((NO_OF_NODES * NO_OF_NODES * 3, num_variables))
    edge_selection_b = torch.zeros((NO_OF_NODES * NO_OF_NODES * 3, 1))
    count = 0
    for i in range(0, NO_OF_NODES):
        for j in range(0, NO_OF_NODES):
            if i != j:
                edge_selection_A[count, i] = -1
                edge_selection_A[count, NO_OF_NODES + i * NO_OF_NODES + j] = 1
                count += 1
                edge_selection_A[count, i] = -1
                edge_selection_A[count, NO_OF_NODES + j * NO_OF_NODES + i] = 1
                count += 1
                edge_selection_A[count, i] = 1
                edge_selection_A[count, j] = 1
                edge_selection_A[count, NO_OF_NODES + i * NO_OF_NODES + j] = -1
                edge_selection_b[count] = -1
                count += 1
    # Abstract fact select
    abstract_fact_select_A = torch.cat(
        (torch.tensor(is_abstract_fact), torch.zeros((NO_OF_NODES * NO_OF_NODES)))
    ).unsqueeze(0)
    abstract_fact_select_b = torch.Tensor([[abstract_limit]]) * -1

    # Combine data
    edge_weights = np.triu(inter_edges_weights) + outgoing_edges_weigths.T
    A = torch.cat(
        (
            root_constraint_A,
            incoming_constraints_A,
            edge_selection_A,
            abstract_fact_select_A,
        )
    )
    b = torch.cat(
        (
            root_constraint_b,
            incoming_constraints_b,
            edge_selection_b,
            abstract_fact_select_b,
        )
    )
    node_scores, edge_scores = torch.ones((NO_OF_NODES)), torch.Tensor(
        edge_weights
    ).reshape(NO_OF_NODES * NO_OF_NODES)
    c = torch.cat((node_scores, edge_scores)) * -1
    constraints = torch.cat((A, b), dim=1)

    return constraints, c


class CVXPYPocTest(unittest.TestCase):
    # @unittest.skip
    def test_abstract_limit(self):

        NO_OF_NODES = 6
        # Testing numpy
        # temp_edge = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        # nodes = np.array([1, 1, 0])
        # print((temp_edge - nodes).reshape(-1))

        num_variables = NO_OF_NODES + NO_OF_NODES * NO_OF_NODES
        is_abstract_fact = np.array([0, 1, 1, 1, 1, 1])
        abstract_limit = 1

        ####################################################
        outgoing_edges_weigths = np.array(
            [
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        inter_edges_weights = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        is_abstract_fact = np.array([0, 1, 1, 1, 1, 1])
        abstract_limit = 1

        ####################################################

        incoming_edges = np.where(
            outgoing_edges_weigths.T != 0,
            np.ones_like(outgoing_edges_weigths),
            np.zeros_like(outgoing_edges_weigths),
        )
        inter_edges = np.where(
            inter_edges_weights != 0,
            np.ones_like(inter_edges_weights),
            np.zeros_like(inter_edges_weights),
        )
        edge_weights = np.triu(inter_edges_weights) + outgoing_edges_weigths.T

        constraints, c = build_constraints(
            NO_OF_NODES,
            is_abstract_fact,
            abstract_limit,
            inter_edges_weights,
            outgoing_edges_weigths,
        )

        constraints = constraints.unsqueeze(0)
        c = c.unsqueeze(0)

        comboptnet = CombOptNetModule({"lb": 0, "ub": 1}, tau=0.5)

        c = torch.randn(1, 42, requires_grad=True)

        results = comboptnet(c, constraints)[0]
        print(results.sum().backward())

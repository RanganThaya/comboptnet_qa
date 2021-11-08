import unittest

import cvxpy as cp
import numpy as np
import torch
from comboptnet_qa.models.comboptnet.comboptnet import CombOptNetModule
from cvxpylayers.torch import CvxpyLayer
from dynaconf import settings

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


class CVXPYPocTest(unittest.TestCase):
    #     @unittest.skip
    def test_root_constraint(self):
        nodes = cp.Variable((NO_OF_NODES), boolean=True)
        objective = cp.Minimize(cp.sum(nodes))
        constraints = [nodes[1] == 1]
        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(prob.value - 1.0) < 0.00001
        # cvxpylayer = CvxpyLayer(prob, parameters=[], variables=[nodes])

    @unittest.skip
    def test_chaining_constraint(self):
        outgoing_edges = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 1],
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
        incoming_edges = outgoing_edges.T

        nodes = cp.Variable((NO_OF_NODES), boolean=True)
        incoming_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=True)

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        objective = cp.Maximize(cp.sum(nodes))
        constraints = [nodes[0] == 1, nodes - incoming_edges_param @ nodes <= 0]
        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()
        incoming_edges_param.value = incoming_edges
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        print(prob.status)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 9.0) < 0.00001
        assert ((nodes.value) == np.ones((NO_OF_NODES))).all()

        outgoing_edges = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        incoming_edges = outgoing_edges.T
        incoming_edges_param.value = incoming_edges
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 8.0) < 0.00001
        assert abs(nodes.value[8] - 0.0) < 0.00001

        outgoing_edges = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        incoming_edges = outgoing_edges.T
        incoming_edges_param.value = incoming_edges
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 7.0) < 0.00001
        assert abs(nodes.value[8] - 0.0) < 0.00001
        assert abs(nodes.value[7] - 0.0) < 0.00001

        outgoing_edges = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        incoming_edges = outgoing_edges.T
        incoming_edges_param.value = incoming_edges
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 7.0) < 0.00001
        assert abs(nodes.value[8] - 0.0) < 0.00001
        assert abs(nodes.value[7] - 0.0) < 0.00001

        outgoing_edges = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        incoming_edges = outgoing_edges.T
        incoming_edges_param.value = incoming_edges
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 8.0) < 0.00001
        assert abs(nodes.value[7] - 0.0) < 0.00001

        outgoing_edges = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        incoming_edges = outgoing_edges.T
        incoming_edges_param.value = incoming_edges
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 8.0) < 0.00001
        assert abs(nodes.value[7] - 0.0) < 0.00001

        outgoing_edges = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        incoming_edges = outgoing_edges.T
        incoming_edges_param.value = incoming_edges
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 9.0) < 0.00001

    @unittest.skip
    def test_edge_selection_constraint(self):

        NO_OF_NODES = 3
        # Testing numpy
        # temp_edge = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        # nodes = np.array([1, 1, 0])
        # print((temp_edge - nodes).reshape(-1))

        nodes = cp.Variable((NO_OF_NODES), boolean=True)
        edges = cp.Variable((NO_OF_NODES, NO_OF_NODES), boolean=True)
        inter_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=True)
        incoming_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=True)
        edge_weights_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES))

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        objective = cp.Maximize(
            cp.sum(nodes)
            + cp.sum(
                cp.reshape(
                    cp.multiply(edge_weights_param, edges), (NO_OF_NODES * NO_OF_NODES)
                )
            )
        )
        constraints = [
            nodes[0] == 1,
            nodes - incoming_edges_param @ nodes <= 0,
            cp.multiply(
                inter_edges_param,
                edges - np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - cp.transpose(
                    (np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)))
                ),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - (
                    np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES))
                    + cp.transpose(
                        (
                            np.ones((NO_OF_NODES, 1))
                            @ cp.reshape(nodes, (1, NO_OF_NODES))
                        )
                    )
                    - np.ones((NO_OF_NODES, NO_OF_NODES))
                ),
            )
            >= 0,
        ]
        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()

        outgoing_edges = np.array(
            [
                [1, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        incoming_edges = outgoing_edges.T
        inter_edges = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        edge_weights = np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]])

        incoming_edges_param.value = incoming_edges
        inter_edges_param.value = np.triu(inter_edges)
        edge_weights_param.value = edge_weights
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 8.0) < 0.00001

        outgoing_edges_weigths = np.array(
            [
                [1, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        incoming_edges = np.where(
            outgoing_edges_weigths.T != 0,
            np.ones_like(outgoing_edges_weigths),
            np.zeros_like(outgoing_edges_weigths),
        )
        inter_edges_weights = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
        inter_edges = np.where(
            inter_edges_weights != 0,
            np.ones_like(inter_edges_weights),
            np.zeros_like(inter_edges_weights),
        )
        edge_weights = np.triu(inter_edges_weights) + outgoing_edges_weigths.T

        incoming_edges_param.value = incoming_edges
        print(np.triu(inter_edges) + incoming_edges)
        inter_edges_param.value = np.triu(inter_edges) + incoming_edges
        edge_weights_param.value = edge_weights
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 5.0) < 0.00001

        outgoing_edges_weigths = np.array(
            [
                [1, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        incoming_edges = np.where(
            outgoing_edges_weigths.T != 0,
            np.ones_like(outgoing_edges_weigths),
            np.zeros_like(outgoing_edges_weigths),
        )
        inter_edges_weights = np.array([[0, 0, 0], [0, 0, -4], [0, -4, 0]])
        inter_edges = np.where(
            inter_edges_weights != 0,
            np.ones_like(inter_edges_weights),
            np.zeros_like(inter_edges_weights),
        )
        edge_weights = np.triu(inter_edges_weights) + outgoing_edges_weigths.T

        incoming_edges_param.value = incoming_edges
        inter_edges_param.value = np.triu(inter_edges) + incoming_edges
        edge_weights_param.value = edge_weights
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(result - 4.0) < 0.00001

        # NOTE: Remember to reduce 2 (Question node and Question edge)

    @unittest.skip
    def test_abstract_limit(self):

        NO_OF_NODES = 6
        # Testing numpy
        # temp_edge = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        # nodes = np.array([1, 1, 0])
        # print((temp_edge - nodes).reshape(-1))

        nodes = cp.Variable((NO_OF_NODES), boolean=True)
        edges = cp.Variable((NO_OF_NODES, NO_OF_NODES), boolean=True)
        inter_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=True)
        incoming_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=True)
        edge_weights_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES))
        is_abstract_fact_param = cp.Parameter((NO_OF_NODES))
        abstract_limit_value_param = cp.Parameter()

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        objective = cp.Maximize(
            cp.sum(nodes)
            + cp.sum(
                cp.reshape(
                    cp.multiply(edge_weights_param, edges), (NO_OF_NODES * NO_OF_NODES)
                )
            )
        )
        constraints = [
            nodes[0] == 1,
            nodes - incoming_edges_param @ nodes <= 0,
            cp.multiply(
                inter_edges_param,
                edges - np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - cp.transpose(
                    (np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)))
                ),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - (
                    np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES))
                    + cp.transpose(
                        (
                            np.ones((NO_OF_NODES, 1))
                            @ cp.reshape(nodes, (1, NO_OF_NODES))
                        )
                    )
                    - np.ones((NO_OF_NODES, NO_OF_NODES))
                ),
            )
            >= 0,
            cp.sum(cp.multiply(is_abstract_fact_param, nodes))
            <= abstract_limit_value_param,
        ]
        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()

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
        is_abstract_fact = np.array([0, 0, 0, 0, 1, 1])
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
        is_abstract_fact_param.value = is_abstract_fact
        abstract_limit_value_param.value = abstract_limit

        incoming_edges_param.value = incoming_edges
        inter_edges_param.value = np.triu(inter_edges) + incoming_edges
        edge_weights_param.value = edge_weights
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(np.sum(nodes.value) - 5.0) < 0.0001

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
                [0, 0, -1, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        is_abstract_fact = np.array([0, 0, 0, 0, 1, 1])
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
        is_abstract_fact_param.value = is_abstract_fact
        abstract_limit_value_param.value = abstract_limit

        incoming_edges_param.value = incoming_edges
        inter_edges_param.value = np.triu(inter_edges) + incoming_edges
        edge_weights_param.value = edge_weights
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(np.sum(nodes.value) - 5.0) < 0.0001
        assert abs(result - 10.0) < 0.0001

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
                [0, 0, -4, 0, 0, 0],
                [0, -4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        is_abstract_fact = np.array([0, 0, 0, 0, 1, 1])
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
        is_abstract_fact_param.value = is_abstract_fact
        abstract_limit_value_param.value = abstract_limit

        incoming_edges_param.value = incoming_edges
        inter_edges_param.value = np.triu(inter_edges) + incoming_edges
        edge_weights_param.value = edge_weights
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(np.sum(nodes.value) - 4.0) < 0.0001
        assert abs(result - 8.0) < 0.0001

    @unittest.skip
    def test_grounding_neighbor(self):

        NO_OF_NODES = 6
        # Testing numpy
        # temp_edge = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        # nodes = np.array([1, 1, 0])
        # print((temp_edge - nodes).reshape(-1))

        nodes = cp.Variable((NO_OF_NODES), boolean=True)
        edges = cp.Variable((NO_OF_NODES, NO_OF_NODES), boolean=True)
        inter_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=True)
        incoming_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=True)
        edge_weights_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES))
        is_abstract_fact_param = cp.Parameter((NO_OF_NODES))
        abstract_limit_value_param = cp.Parameter()

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        objective = cp.Maximize(
            cp.sum(nodes)
            + cp.sum(
                cp.reshape(
                    cp.multiply(edge_weights_param, edges), (NO_OF_NODES * NO_OF_NODES)
                )
            )
        )
        constraints = [
            nodes[0] == 1,
            nodes - incoming_edges_param @ nodes <= 0,
            cp.multiply(
                inter_edges_param,
                edges - np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - cp.transpose(
                    (np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)))
                ),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - (
                    np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES))
                    + cp.transpose(
                        (
                            np.ones((NO_OF_NODES, 1))
                            @ cp.reshape(nodes, (1, NO_OF_NODES))
                        )
                    )
                    - np.ones((NO_OF_NODES, NO_OF_NODES))
                ),
            )
            >= 0,
            cp.sum(cp.multiply(is_abstract_fact_param, nodes))
            <= abstract_limit_value_param,
            (incoming_edges_param + incoming_edges_param.T) @ nodes
            - 2
            + 2 * (1 - cp.multiply(nodes, 1 - is_abstract_fact_param))
            >= 0,
        ]
        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()
        assert prob.is_dcp(dpp=True)

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
                [0, 0, -4, 0, 0, 0],
                [0, -4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        is_abstract_fact = np.array([0, 0, 0, 0, 1, 1])
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
        is_abstract_fact_param.value = is_abstract_fact
        abstract_limit_value_param.value = abstract_limit

        incoming_edges_param.value = incoming_edges
        inter_edges_param.value = np.triu(inter_edges) + incoming_edges
        edge_weights_param.value = edge_weights
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL
        assert abs(np.sum(nodes.value) - 3.0) < 0.0001
        assert abs(result - 6.0) < 0.0001

        ####################################################
        outgoing_edges_weigths = np.array(
            [
                [1, 20, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        inter_edges_weights = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        is_abstract_fact = np.array([0, 0, 0, 0, 1, 1])
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
        is_abstract_fact_param.value = is_abstract_fact
        abstract_limit_value_param.value = abstract_limit

        incoming_edges_param.value = incoming_edges
        inter_edges_param.value = np.triu(inter_edges) + incoming_edges
        edge_weights_param.value = edge_weights
        result = prob.solve(solver=cp.CPLEX, verbose=True)
        assert prob.status == cp.OPTIMAL

    @unittest.skip
    def test_cvxpy_layers(self):
        NO_OF_NODES = 6
        # Testing numpy
        # temp_edge = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        # nodes = np.array([1, 1, 0])
        # print((temp_edge - nodes).reshape(-1))

        nodes = cp.Variable((NO_OF_NODES), boolean=False, nonneg=True)
        edges = cp.Variable((NO_OF_NODES, NO_OF_NODES), boolean=False, nonneg=True)
        inter_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=False)
        incoming_edges_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES), boolean=False)
        edge_weights_param = cp.Parameter((NO_OF_NODES, NO_OF_NODES))
        is_abstract_fact_param = cp.Parameter((NO_OF_NODES,))
        abstract_limit_value_param = cp.Parameter()

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        # Testing in numpy
        # test = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # print(incoming_edges @ test)

        objective = cp.Maximize(
            cp.sum(nodes)
            + cp.sum(
                cp.reshape(
                    cp.multiply(edge_weights_param, edges), (NO_OF_NODES * NO_OF_NODES)
                )
            )
        )
        constraints = [
            nodes[0] == 1,
            nodes - incoming_edges_param @ nodes <= 0,
            cp.multiply(
                inter_edges_param,
                edges - np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - cp.transpose(
                    (np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES)))
                ),
            )
            <= 0,
            cp.multiply(
                inter_edges_param,
                edges
                - (
                    np.ones((NO_OF_NODES, 1)) @ cp.reshape(nodes, (1, NO_OF_NODES))
                    + cp.transpose(
                        (
                            np.ones((NO_OF_NODES, 1))
                            @ cp.reshape(nodes, (1, NO_OF_NODES))
                        )
                    )
                    - np.ones((NO_OF_NODES, NO_OF_NODES))
                ),
            )
            >= 0,
            cp.sum(cp.multiply(is_abstract_fact_param, nodes)) <= 1,
            (incoming_edges_param + incoming_edges_param.T) @ nodes
            - 2
            + 2 * (1 - cp.multiply(nodes, 1 - is_abstract_fact_param))
            >= 0,
        ]
        prob = cp.Problem(objective, constraints)
        assert prob.is_dpp()
        assert prob.is_dcp(dpp=True)

        cvxpylayer = CvxpyLayer(
            prob,
            parameters=[
                inter_edges_param,
                incoming_edges_param,
                edge_weights_param,
                is_abstract_fact_param,
            ],
            variables=[nodes, edges],
        )

        ####################################################
        outgoing_edges_weigths = np.array(
            [
                [1, 20, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        inter_edges_weights = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        is_abstract_fact = np.array([0, 0, 0, 0, 1, 1])

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

        # solve the problem
        solution = cvxpylayer(
            torch.tensor(
                np.triu(inter_edges) + incoming_edges,
                requires_grad=True,
                dtype=torch.float64,
            ),
            torch.tensor(incoming_edges, requires_grad=True, dtype=torch.float64),
            torch.tensor(edge_weights, requires_grad=True, dtype=torch.float64),
            torch.tensor(is_abstract_fact, requires_grad=True, dtype=torch.float64),
        )

import unittest

import numpy as np
import sympy


class CVXPYPocTest(unittest.TestCase):
    def test_linear_independence(self):
        num_nodes = 5
        num_choices = 2

        matrix = np.zeros((num_nodes * num_nodes, num_nodes * num_nodes))
        matrix = np.eye(num_nodes * num_nodes)
        # matrix = np.zeros((num_variables, num_variables))

        # for f_nodes in range(num_choices, num_nodes):
        # matrix[f_nodes, f_nodes] = 1
        count = 0
        registered = set()
        for i in range(num_nodes * num_nodes):
            row = int(i / num_nodes)
            col = i % num_nodes
            if not (row, col) in registered and not (col, row) in registered:
                matrix[count, row * num_nodes + col] = 1
                matrix[count, col * num_nodes + row] = 1
                matrix[count, row * num_nodes + row] = 1
                matrix[count, col * num_nodes + col] = 1
                registered.add((row, col))
                registered.add((col, row))
                count += 1

        _, indexes = sympy.Matrix(matrix).T.rref()  # T is for transpose
        print(indexes)

        # matrix[4, num_nodes + ]
        lambdas, V = np.linalg.eig(matrix.T)
        # print(matrix[lambdas == 0, :])

        matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])

        _, indexes = sympy.Matrix(matrix).T.rref()  # T is for transpose
        # print(indexes)

        # The linearly dependent row vectors
        # print(matrix[lambdas == 0, :])

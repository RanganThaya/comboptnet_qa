from unittest import TestCase

import cvxpy as cp
import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
from diffcp.solver import solve_batch


class PSDMapTest(TestCase):
    def test_psd_map(self):

        num_nodes = 5
        edges = cp.Variable((num_nodes, num_nodes))

        C = cp.reshape(cp.hstack((np.ones((1)), cp.diag(edges).T)), (1, num_nodes + 1))
        C2 = cp.hstack((cp.reshape(cp.diag(edges), (num_nodes, 1)), edges))
        C3 = cp.vstack((C, C2))

        edge_weight_param = cp.Parameter((num_nodes, num_nodes))
        solved_edges_param = cp.Parameter((num_nodes, num_nodes))
        constraints = [
            C3 >> 0,
            C3.T == C3,
            cp.sum(cp.diag(edges)) <= 3,
            cp.sum(cp.diag(edges)[:2]) == 1,
            # cp.norm(edges - solved_edges_param, 2) <= 1,
        ]
        obj = cp.Minimize(cp.norm(edges - solved_edges_param, "fro"))
        # obj = cp.Maximize(cp.sum(cp.multiply(edges, edge_weight_param)))
        prob = cp.Problem(obj, constraints)
        ilp_prob = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(edges, edge_weight_param))), constraints
        )

        data, _, _ = prob.get_problem_data(solver=cp.SCS)
        compiler = data[cp.settings.PARAM_PROB]
        param_ids = [p.id for p in [solved_edges_param]]
        cone_dims = dims_to_solver_dict(data["dims"])

        ilp_data, _, _ = ilp_prob.get_problem_data(solver=cp.SCS)
        ilp_compiler = ilp_data[cp.settings.PARAM_PROB]
        ilp_param_ids = [p.id for p in [edge_weight_param]]
        ilp_cone_dims = dims_to_solver_dict(ilp_data["dims"])

        edge_weights = np.random.rand(num_nodes, num_nodes)

        As, bs, cs, cone_dicts = [], [], [], []
        c, _, neg_A, b = ilp_compiler.apply_parameters(
            dict(zip(ilp_param_ids, [edge_weights])), keep_zeros=True
        )
        A = -neg_A  # cvxpy canonicalizes -A
        As.append(A)
        bs.append(b)
        cs.append(c)
        cone_dicts.append(ilp_cone_dims)
        xs, ys, ss = solve_batch(As, bs, cs, cone_dicts, **{})

        As, bs, cs, cone_dicts = [], [], [], []
        c, _, neg_A, b = ilp_compiler.apply_parameters(
            dict(zip(ilp_param_ids, [edge_weights])), keep_zeros=True
        )
        A = -neg_A  # cvxpy canonicalizes -A
        As.append(A)
        bs.append(b)
        cs.append(c)
        cone_dicts.append(ilp_cone_dims)
        xs, ys, ss = solve_batch(As, bs, cs, cone_dicts, **{})

        variables = [edges]
        var_dict = {v.id for v in variables}

        sol = [[] for _ in range(len(variables))]

        sltn_dict = ilp_compiler.split_solution(xs[0], active_vars=var_dict)
        for j, v in enumerate(variables):
            sol[j].append(sltn_dict[v.id])

        edge_sol = sol[0][0]

        As_p, bs_p, cs_p, cone_dicts_p = [], [], [], []
        c, _, neg_A, b = compiler.apply_parameters(
            dict(zip(param_ids, [edge_sol])), keep_zeros=True
        )
        A = -neg_A  # cvxpy canonicalizes -A
        As_p.append(A)
        bs_p.append(b)
        cs_p.append(c)
        cone_dicts_p.append(cone_dims)
        xs_p, ys_p, ss_p = solve_batch(As_p, bs_p, cs_p, cone_dicts_p, **{})

        new_ss = bs[0] - As[0] @ xs_p[0][1:]

        A_t = As[0].T
        # print(A_t.shape)
        # print(cs[0].shape)

        a = As[0].toarray().T
        b = -1 * cs[0]

        y_var = cp.Variable((bs[0].shape))
        new_obj = cp.Minimize(bs[0].T @ y_var)
        constraints = [As[0].T @ y_var + cs[0] == 0, ss[0].T @ y_var == 0]
        prob = cp.Problem(new_obj, constraints)
        prob.solve(solver=cp.SCS)
        print(prob.status)
        print(y_var.value)
        print(ys[0])

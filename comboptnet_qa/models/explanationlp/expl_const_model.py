from platform import node
from selectors import EpollSelector
from tkinter import dialog
from xmlrpc.client import boolean

import cvxpy as cp
import numpy as np
import torch
from comboptnet_qa.models.base_clamp import (HalfClamp, NegClamp, PosClamp,
                                             QuestionClamp)
from comboptnet_qa.models.comboptnet.blackbox_ilp import BlackBoxILP
from comboptnet_qa.models.comboptnet.comboptnet import CombOptNetModule
from comboptnet_qa.models.comboptnet.utils import torch_parameter_from_numpy
from comboptnet_qa.models.comboptnet_pytorch import CombOptNet
# from comboptnet_qa.models.cvxpy_model.cvxpy_ilp import CvxpyLayer
from cvxpylayers.torch import CvxpyLayer
# from comboptnet_qa.models.cvxpy_model.cvxpy_ilp import
from loguru import logger
from torch import nn
from transformers import AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


class ExplanationLPModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        # emb_wi,
        transformer_model,
        hyp_max_len,
        fact_max_len,
        disable_transformer,
        only_answer,
        # w_embd_size=1024,
        w_embd_size=768,
        opts={},
        num_choices=4,
        num_facts=3,
    ):
        super(ExplanationLPModel, self).__init__()
        self.num_choices = num_choices
        self.num_nodes = num_nodes + self.num_choices
        self.opts = opts

        num_nodes = num_nodes + self.num_choices

        edges = cp.Variable((num_nodes, num_nodes))
        C = cp.reshape(cp.hstack((np.ones((1)), cp.diag(edges).T)), (1, num_nodes + 1))
        C2 = cp.hstack((cp.reshape(cp.diag(edges), (num_nodes, 1)), edges))
        C3 = cp.vstack((C, C2))

        edge_weight_param = cp.Parameter((num_nodes, num_nodes))
        solved_edges_param = cp.Parameter((num_nodes, num_nodes))
        constraints = [
            C3 >> 0,
            C3.T == C3,
            # cp.sum(cp.diag(edges)) <= 3,
            # cp.sum(cp.diag(edges)[: self.num_choices]) == 1,
            # cp.norm(edges - solved_edges_param, 2) <= 1,
        ]
        obj = cp.Minimize(cp.norm(edges - edge_weight_param, 2))
        # obj = cp.Maximize(cp.sum(cp.multiply(edges, edge_weight_param)))
        prob = cp.Problem(obj, constraints)

        # self.comboptnet = CombOptNet()
        # self.comboptnet = CombOptNet(num_workers=8)
        # self.comboptnet = CombOptNet(num_workers=8)
        # self.comboptnet = CombOptNetModule(
        # {"lb": 0, "ub": 1}, tau=0.5, use_canonical_basis=True
        # )
        self.comboptnet = BlackBoxILP(
            {"lb": 0, "ub": 1},
            tau=0.5,
            # lambda_val=50,
            # lambda_val=10,
            lambda_val=opts["lambda_val"],
            num_nodes=num_nodes,
        )
        # self.comboptnet = BlackBoxILP({"lb": 0, "ub": 1}, tau=0.5, lambda_val=5)
        # self.comboptnet = CombOptNetModule(
        #     {"lb": 0, "ub": 1}, tau=0.5, use_canonical_basis=True
        # )
        # self.comboptnet = CombOptNetModule({"lb": 0, "ub": 1}, tau=0.5)
        self.warm_starts = None

        # self.cvxpylayer = CvxpyLayer(
        #     prob,
        #     parameters=[
        #         edge_weight_param,
        #         # solved_edges_param,
        #     ],
        #     variables=[edges],
        #     # ilp_problem=ilp_problem,
        # )

        edge_weight_param = cp.Parameter((num_nodes, num_nodes))
        edges = cp.Variable((num_nodes, num_nodes))
        obj = cp.Minimize(cp.norm(edges - edge_weight_param, 2))
        # prob = cp.Problem(obj, )
        # prob = cp.Problem(obj, [edges >> 0, edges.T == edges])
        prob = cp.Problem(obj, constraints)

        # prob = cp.Problem(obj, [cp.trace(edges) == 1, edges.T == edges, edges >= 0])

        self.cvxpylayer = CvxpyLayer(
            prob,
            parameters=[
                edge_weight_param,
            ],
            variables=[edges],
        )

        logger.info(f"Opts provided {opts}")

        self.w_embd_size = w_embd_size
        self.hyp_max_len = hyp_max_len
        self.fact_max_len = fact_max_len

        self.model = AutoModel.from_pretrained(f"{transformer_model}")

        self.only_answer = only_answer
        if disable_transformer:
            logger.warning("Disable transformer finetuning")
            for params in self.model.parameters():
                params.requires_grad = False

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        clamp = PosClamp()
        question_clamp = QuestionClamp()

        self.question_clamp_param = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        question_clamp.apply(self.question_clamp_param)

        # score = 0.01

        self.abstract_abstract_lexical_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.abstract_abstract_lexical_clamp_param)

        self.abstract_abstract_similarity_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.abstract_abstract_similarity_clamp_param)
        self.grounding_abstract_overlap_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.grounding_abstract_overlap_clamp_param)
        self.question_grounding_overlap_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.question_grounding_overlap_clamp_param)
        self.question_abstract_overlap_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.question_abstract_overlap_clamp_param)
        self.question_abstract_relevance_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.question_abstract_relevance_clamp_param)
        self.question_grounding_relevance_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.question_grounding_relevance_clamp_param)

        self.grounding_grounding_overlap_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.grounding_grounding_overlap_clamp_param)

        self.softmax = nn.Softmax(dim=1)

        self.nll_loss = nn.NLLLoss()
        self.cross_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.nll_loss = nn.NLLLoss()

    def forward(
        self,
        hypothesis_input_ids,
        fact_input_ids,
        hypothesis_attention_mask,
        fact_attention_mask,
        labels,
        abstract_abstract_lexical,
        question_abstract_edges,
        question_grounding_edges,
        grounding_grounding_edges,
        grounding_abstract_edges,
        # A,
        # b,
        constraints,
        is_abstract,
        similarity_scores,
        epoch=0,
        **kwargs,
    ):

        fact_input_ids = fact_input_ids.view(-1, self.fact_max_len)
        fact_attention_mask = fact_attention_mask.view(-1, self.fact_max_len)

        hypothesis_input_ids = hypothesis_input_ids.view(-1, self.hyp_max_len)
        hypothesis_attention_mask = hypothesis_attention_mask.view(-1, self.hyp_max_len)

        fact_seq_embedding = self.model(fact_input_ids, fact_attention_mask)
        fact_seq_embedding = mean_pooling(fact_seq_embedding, fact_attention_mask)
        hypothesis_seq_embedding = self.model(
            hypothesis_input_ids, hypothesis_attention_mask
        )
        hypothesis_seq_embedding = mean_pooling(
            hypothesis_seq_embedding, hypothesis_attention_mask
        )

        hypothesis_seq_embedding = hypothesis_seq_embedding.view(
            -1, self.num_choices, self.w_embd_size
        )
        fact_seq_embedding = fact_seq_embedding.view(
            -1, self.num_nodes - self.num_choices, self.w_embd_size
        )

        hypothesis_norm = (
            hypothesis_seq_embedding / hypothesis_seq_embedding.norm(dim=2)[:, :, None]
        )

        fact_norm = fact_seq_embedding / fact_seq_embedding.norm(dim=2)[:, :, None]

        relevance_scores = torch.bmm(hypothesis_norm, fact_norm.transpose(1, 2))

        question_abstract_similarity = torch.zeros_like(question_abstract_edges)
        question_abstract_similarity[
            :, self.num_choices :, : self.num_choices
        ] = relevance_scores.transpose(1, 2)
        question_abstract_similarity[
            :, : self.num_choices, self.num_choices :
        ] = relevance_scores

        question_abstract_edges = question_abstract_edges + torch.transpose(
            question_abstract_edges, 1, 2
        )
        question_grounding_edges = question_grounding_edges + torch.transpose(
            question_grounding_edges, 1, 2
        )

        grounding_abstract_edges = grounding_abstract_edges + torch.transpose(
            grounding_abstract_edges, 1, 2
        )

        question_abstract_adj = torch.where(
            question_abstract_edges != 0,
            torch.ones_like(question_abstract_similarity),
            torch.zeros_like(question_abstract_similarity),
        )

        question_grounding_adj = torch.where(
            question_grounding_edges != 0,
            torch.ones_like(question_abstract_similarity),
            torch.zeros_like(question_abstract_similarity),
        )

        edge_weights = (
            self.abstract_abstract_lexical_clamp_param * abstract_abstract_lexical * -1
            + self.grounding_grounding_overlap_clamp_param
            * grounding_grounding_edges
            * -1
            + self.question_grounding_overlap_clamp_param * question_grounding_edges
            + self.question_abstract_overlap_clamp_param * question_abstract_edges
            # self.question_abstract_overlap_clamp_param * question_abstract_edges
            + self.grounding_abstract_overlap_clamp_param * grounding_abstract_edges
            # + self.question_grounding_relevance_clamp_param
            # * question_abstract_similarity
            # * (question_grounding_adj)
            + self.question_abstract_relevance_clamp_param
            * question_abstract_similarity
            # question_abstract_similarity
            * (question_abstract_adj)
        )

        # edge_weights = torch.nn.functional.softmax(
        #     edge_weights.reshape(-1, self.num_nodes * self.num_nodes), dim=1
        # )
        edge_weights = edge_weights.reshape(-1, self.num_nodes * self.num_nodes) * -1

        edge_weights = torch.triu(edge_weights, diagonal=1)

        # A, b = constraints[:, :, :-1], constraints[:, :, -1]
        # results = self.comboptnet(A, b, c)[0]
        results = self.comboptnet(edge_weights, constraints)
        # (results, xs, ys, ss) = self.cvxpylayer(
        #     edge_weights,
        #     constraints=constraints,
        #     num_nodes=self.num_nodes,
        #     solver_args={
        #         "warm_starts": self.warm_starts,
        #     },
        # )
        # # # print(results)
        # self.warm_starts = (xs, ys, ss)
        # results = results.view(-1, self.num_nodes, self.num_nodes)
        comb_results = results.view(-1, self.num_nodes, self.num_nodes)

        results = torch.triu(comb_results)

        # upper_tri = (
        #     torch.triu(results, diagonal=1)
        #     * edge_weights.view(-1, self.num_nodes, self.num_nodes)
        #     * -1
        # )
        # upper_sum = torch.sum(upper_tri, dim=2)
        # upper_sum = upper_sum[:, : self.num_choices]

        # upper_bias = torch.sum(upper_sum[:, self.num_choices :], dim=1)
        # upper_bias = upper_bias.unsqueeze(1).repeat(1, self.num_choices)
        # upper_sum = upper_sum + upper_bias
        # print(upper_sum)

        optimizer_scores = (
            results * edge_weights.view(-1, self.num_nodes, self.num_nodes) * -1
        )
        summed_up = torch.sum(
            optimizer_scores.view(-1, self.num_nodes * self.num_nodes), dim=1
        )

        predictions = torch.diagonal(results, dim1=1, dim2=2)
        comb_predictions = torch.diagonal(comb_results, dim1=1, dim2=2)

        # print(torch.linalg.matrix_rank(torch.round(results)))

        # U, S, Vh = torch.svd_lowrank(results, q=2, niter=10)

        # U = U[:, :, 0]
        # print(U)
        # S = torch.sqrt(S[:, 0]).unsqueeze(1).repeat(1, self.num_nodes)
        # U = U * S

        # svd_answer_predicts = torch.abs(U[:, : self.num_choices])

        real_answer_predicts = predictions[:, : self.num_choices]
        summed_up = summed_up.unsqueeze(1).repeat(1, self.num_choices)

        # print(summed_up)
        # print(real_answer_predicts)

        real_answer_predicts = (
            real_answer_predicts * summed_up
            + ((1 - real_answer_predicts) * summed_up) * -1
        )

        # real_answer_predicts = real_answer_predicts * summed_up
        # sigmoid_predicts = self.sigmoid(real_answer_predicts)

        # pos_loss = (1 - (torch.masked_select(sigmoid_predicts, labels.bool()))).mean()
        # neg_loss = ((torch.masked_select(sigmoid_predicts, (1 - labels).bool()))).mean()
        # answer_loss = pos_loss + neg_loss
        # print(real_answer_predicts)
        # real_answer_predicts = real_answer_predicts * self.opts.get("temperature", 0.1)
        # print(real_answer_predicts)

        # (psd_edges, xs, ys, ss) = self.cvxpylayer(
        #     results,
        #     solver_args={
        #         "warm_starts": self.warm_starts,
        #     },
        # )
        # self.warm_starts = (xs, ys, ss)

        # print(psd_edges)

        # L = torch.linalg.cholesky(psd_edges)
        # print(L)

        # print(results[0].nonzero())
        # print(optimized_scores)
        # print(optimized_scores.shape)
        # print(similarity_scores[0])
        # print(similarity_scores.shape)

        # Answer loss

        # # Explanation loss
        # print(results.shape)
        # print(torch.argmax(labels, dim=1))
        selection_indices = torch.argmax(labels, dim=1).unsqueeze(1)
        fact_prediction = results[
            torch.arange(results.shape[0])[:, None], selection_indices, :
        ][:, :, self.num_choices :].squeeze(1)

        exp_loss = (
            1 - (torch.masked_select(fact_prediction, similarity_scores.bool()))
        ).mean()

        # real_answer_predicts[real_answer_predicts == 0] = -6

        # loss = pos_loss + neg_loss
        loss = (
            self.cross_loss(real_answer_predicts, torch.argmax(labels, dim=1))
            + exp_loss
        )
        # loss = (real_answer_predicts * (1 - labels)).mean()

        # print(upper_sum)
        # print(labels)
        # loss = (
        #     self.cross_loss(upper_sum, torch.argmax(labels, dim=1))
        #     # + exp_loss
        # )
        # loss = self.kl_loss(
        #     torch.nn.functional.log_softmax(upper_sum, dim=1),
        #     torch.nn.functional.log_softmax(labels, dim=1),
        # )
        # loss = answer_loss
        # loss = torch.abs(real_answer_predicts - labels * summed_up).mean()
        # loss = ((1 - labels) * real_answer_predicts).mean()
        return (
            # loss + 0.01 * learning_reg,
            loss,
            # svd_answer_predicts,
            # upper_sum,
            real_answer_predicts,
            # predictions[:, : self.num_choices],
            # fact_prediction,
            comb_predictions[:, self.num_choices :]
            * is_abstract[:, self.num_choices :],
        )

from platform import node
from selectors import EpollSelector
from xmlrpc.client import boolean

import cvxpy as cp
import numpy as np
import torch
from comboptnet_qa.models.base_clamp import HalfClamp, NegClamp, PosClamp, QuestionClamp
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
            lambda_val=10,
            # lambda_val=opts["lambda_val"],
            num_nodes=num_nodes,
        )
        # self.comboptnet = BlackBoxILP({"lb": 0, "ub": 1}, tau=0.5, lambda_val=5)
        # self.comboptnet = CombOptNetModule({"lb": 0, "ub": 1}, tau=0.5)
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

        self.grounding_grounding_overlap_clamp_param = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        clamp.apply(self.grounding_grounding_overlap_clamp_param)

        self.softmax = nn.Softmax(dim=1)

        self.nll_loss = nn.NLLLoss()
        self.cross_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.loss = nn.BCELoss()

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

        edge_weights = (
            # self.abstract_abstract_lexical_clamp_param * abstract_abstract_lexical * -1
            self.grounding_grounding_overlap_clamp_param
            * grounding_grounding_edges
            * -1
            + self.question_grounding_overlap_clamp_param * question_grounding_edges
            + self.question_abstract_overlap_clamp_param * question_abstract_edges
            # self.question_abstract_overlap_clamp_param * question_abstract_edges
            + self.grounding_abstract_overlap_clamp_param * grounding_abstract_edges
            + self.question_abstract_relevance_clamp_param
            * question_abstract_similarity
            # question_abstract_similarity
            # * (question_abstract_adj)
        )

        edge_weights = edge_weights.reshape(-1, self.num_nodes * self.num_nodes) * -1

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
        results = results.view(-1, self.num_nodes, self.num_nodes)

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

        predictions = torch.diagonal(results, dim1=1, dim2=2)

        real_answer_predicts = predictions[:, : self.num_choices]

        # Answer loss
        pos_loss = (
            1 - (torch.masked_select(real_answer_predicts, labels.bool()))
        ).mean()
        neg_loss = (
            (torch.masked_select(real_answer_predicts, (1 - labels).bool()))
        ).mean()
        answer_loss = pos_loss + neg_loss

        # Explanation loss
        fact_prediction = torch.clamp(
            predictions[:, self.num_choices :], min=0.01, max=0.99
        )
        # fact_loss = self.loss(
        #     fact_prediction * is_abstract[:, self.num_choices :],
        #     (similarity_scores * is_abstract[:, self.num_choices :]).double(),
        # )

        # if not self.only_answer:
        # loss = answer_loss + fact_loss
        # else:
        # loss = answer_loss
        # results = torch.clamp(results, min=0.01, max=0.99)
        # results = results.clamp()

        # loss = self.loss(results, similarity_scores.double())

        exp_loss = (
            1
            - (torch.masked_select(results.view(-1), similarity_scores.view(-1).bool()))
        ).mean()
        loss = exp_loss
        # print(similarity_scores.double().nonzero())

        # loss = answer_loss + fact_loss

        clamp_val = (
            self.grounding_grounding_overlap_clamp_param
            + self.question_grounding_overlap_clamp_param
            + self.question_abstract_overlap_clamp_param
            + self.grounding_abstract_overlap_clamp_param
            + self.question_abstract_relevance_clamp_param
        )

        # loss = self.cross_loss(
        #     real_answer_predicts,
        #     torch.argmax(labels.double(), dim=1),
        # )
        return (
            # loss + 0.01 * learning_reg,
            loss,
            real_answer_predicts,
            # predictions[:, : self.num_choices],
            fact_prediction * is_abstract[:, self.num_choices :],
        )

from platform import node
from selectors import EpollSelector
from tokenize import group
from xmlrpc.client import boolean

import cvxpy as cp
import numpy as np
import torch
from comboptnet_qa.models.base_clamp import HalfClamp, NegClamp, PosClamp, QuestionClamp
from comboptnet_qa.models.comboptnet.blackbox_ilp import BlackBoxILP
from comboptnet_qa.models.comboptnet.comboptnet import CombOptNetModule
from comboptnet_qa.models.comboptnet.utils import torch_parameter_from_numpy
from comboptnet_qa.models.comboptnet_pytorch import CombOptNet
from comboptnet_qa.models.cvxpy_model.cvxpy_ilp import CvxpyLayer

# from comboptnet_qa.models.cvxpy_model.cvxpy_ilp import
from loguru import logger
from torch import nn
from transformers import AutoModel


class MPNetClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to BERT's [CLS] token)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


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


class SingleModel(nn.Module):
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
        super(SingleModel, self).__init__()
        self.num_choices = num_choices
        self.num_nodes = num_nodes + 1

        num_nodes = num_nodes + 1

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
            cp.sum(cp.diag(edges)[: self.num_choices]) == 1,
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
            # {"lb": 0, "ub": 1}, tau=0.5, lambda_val=opts["lambda_val"]
            {"lb": 0, "ub": 1},
            tau=0.5,
            lambda_val=100,
        )
        # self.comboptnet = BlackBoxILP({"lb": 0, "ub": 1}, tau=0.5, lambda_val=5)
        # self.comboptnet = CombOptNetModule({"lb": 0, "ub": 1}, tau=0.5)
        # self.comboptnet = CombOptNetModule({"lb": 0, "ub": 1}, tau=0.5)
        self.warm_starts = None

        self.cvxpylayer = CvxpyLayer(
            prob,
            parameters=[
                edge_weight_param,
                # solved_edges_param,
            ],
            variables=[edges],
            # ilp_problem=ilp_problem,
        )

        logger.info(f"Opts provided {opts}")

        self.w_embd_size = w_embd_size
        self.hyp_max_len = hyp_max_len
        self.fact_max_len = fact_max_len

        self.model = AutoModel.from_pretrained(f"{transformer_model}")
        # self.classifier = MPNetClassificationHead(self.w_embd_size)
        # self.answer_model = AutoModel.from_pretrained("microsoft/mpnet-base")

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

        self.dense = nn.Linear(2 * self.w_embd_size, 1)

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
        self.m = nn.LogSoftmax(dim=1)
        self.classifier = nn.Linear(self.num_choices, self.num_choices)
        self.margin_loss = nn.MarginRankingLoss()

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
        eps=1e-8,
        **kwargs,
    ):

        # print(fact_attention_mask[0])
        # print(hypothesis_input_ids, "hypothesis")
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
            -1, self.num_choices * (self.num_nodes - 1), self.w_embd_size
        )

        # hypothesis_norm = hypothesis_seq_embedding / hypothesis_seq_embedding.norm(
        #     dim=2
        # )[:, :, None].clamp(min=eps)
        hypothesis_norm = nn.functional.normalize(hypothesis_seq_embedding, dim=2)
        fact_norm = nn.functional.normalize(fact_seq_embedding, dim=2)

        # fact_norm = fact_seq_embedding / fact_seq_embedding.norm(dim=2)[
        #     :, :, None
        # ].clamp(min=eps)

        # hypothesis_norm = hypothesis_norm.unsqueeze(2)
        fact_norm = fact_norm.view(
            -1, self.num_choices, self.num_nodes - 1, self.w_embd_size
        )

        relevance_scores = torch.einsum("ijk,ijlk->ijl", hypothesis_norm, fact_norm)

        relevance_scores = relevance_scores.view(-1, self.num_nodes - 1)

        print(fact_norm[0])
        print(hypothesis_norm[0])
        print(relevance_scores)

        # print(relevance_scores)
        # print(question_abstract_edges[:, 1:, :1])

        question_abstract_edges = question_abstract_edges.view(
            -1, self.num_nodes, self.num_nodes
        )
        abstract_abstract_lexical = abstract_abstract_lexical.view(
            -1, self.num_nodes, self.num_nodes
        )
        question_grounding_edges = question_grounding_edges.view(
            -1, self.num_nodes, self.num_nodes
        )
        grounding_grounding_edges = grounding_grounding_edges.view(
            -1, self.num_nodes, self.num_nodes
        )
        grounding_abstract_edges = grounding_abstract_edges.view(
            -1, self.num_nodes, self.num_nodes
        )
        constraints = constraints.view(
            -1,
            self.num_nodes * self.num_nodes * 3 + 1 + 1,
            self.num_nodes * self.num_nodes + 1,
        )

        # print(hypothesis_norm.shape)
        # print(fact_norm.shape)

        question_abstract_similarity = torch.zeros_like(question_abstract_edges)
        relevance_scores = relevance_scores.unsqueeze(1)

        question_abstract_similarity[:, 1:, :1] = relevance_scores.transpose(1, 2)
        question_abstract_similarity[:, :1, 1:] = relevance_scores

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

        # print(question_abstract_edges[0])

        edge_weights = (
            self.abstract_abstract_lexical_clamp_param * abstract_abstract_lexical * -1
            #     self.abstract_abstract_lexical_clamp_param
            #     * grounding_grounding_edges
            #     * -1
            #     + self.question_grounding_overlap_clamp_param * question_grounding_edges
            + self.question_abstract_overlap_clamp_param * question_abstract_edges
            # self.question_abstract_overlap_clamp_param * question_abstract_edges
            #     + self.grounding_abstract_overlap_clamp_param * grounding_abstract_edges
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
        results = results.view(-1, self.num_nodes, self.num_nodes)
        # print(results[0])
        #
        results = results.view(-1, self.num_nodes * self.num_nodes)

        # print(edge_weights.view(-1, self.num_nodes, self.num_nodes)[0])

        edge_weights = edge_weights * -1
        scores = torch.sum(edge_weights * results, dim=1)
        scores = scores.view(-1, self.num_choices) * 0.1
        # print(scores)
        # print(labels)
        # scores = self.classifier(scores)
        # scores = scores.squeeze()
        loss = self.cross_loss(scores, torch.argmax(labels, axis=1))
        # predictions = torch.diagonal(results, dim1=1, dim2=2)[:, 1:]
        # predictions = predictions.unsqueeze(2).repeat(1, 1, self.w_embd_size)

        # fact_input_ids = fact_input_ids.view(-1, self.num_nodes - 1, self.fact_max_len)
        # fact_attention_mask = fact_attention_mask.view(
        #     -1, self.num_nodes - 1, self.fact_max_len
        # )
        # fact_attention_mask = fact_attention_mask * predictions

        # fact_input_ids = fact_input_ids.view(
        #     -1, (self.num_nodes - 1) * self.fact_max_len
        # )
        # fact_attention_mask = fact_attention_mask.view(
        #     -1, (self.num_nodes - 1) * self.fact_max_len
        # )

        # sequence_input_ids = torch.cat((hypothesis_input_ids, fact_input_ids), axis=1)
        # sequence_attention_mask = torch.cat(
        #     (hypothesis_attention_mask, fact_attention_mask), axis=1
        # )
        # outputs = self.answer_model(
        #     sequence_input_ids,
        #     attention_mask=sequence_attention_mask,
        # )
        # sequence_output = outputs[0]
        # scores = self.classifier(sequence_output)
        # scores = scores.squeeze()
        # scores = scores.view(-1, self.num_choices)
        # loss = self.cross_loss(scores, torch.argmax(labels, axis=1))

        # print(hypothesis_input_ids)
        # fact_seq_embedding = (
        #     fact_seq_embedding.view(-1, self.num_nodes - 1, self.w_embd_size)
        #     * predictions
        # )
        # fact_seq_embedding = torch.sum(fact_seq_embedding, 1)
        # hypothesis_seq_embedding = hypothesis_seq_embedding.view(-1, self.w_embd_size)

        # combined = torch.cat((fact_seq_embedding, hypothesis_seq_embedding), dim=1)
        # scores = self.dense(combined)
        # scores = scores.squeeze()
        # scores = scores.view(-1, self.num_choices)
        # loss = self.cross_loss(scores, torch.argmax(labels, axis=1))

        # hypothesis_norm = (
        #     hypothesis_seq_embedding / hypothesis_seq_embedding.norm(dim=1)[:, None]
        # )
        # fact_norm = fact_seq_embedding / fact_seq_embedding.norm(dim=1)[:, None]
        # hypothesis_norm = nn.functional.normalize(hypothesis_seq_embedding, dim=1)
        # fact_norm = nn.functional.normalize(fact_seq_embedding, dim=1)

        # scores = torch.einsum("ij,ij->i", hypothesis_norm.double(), fact_norm)
        # scores = scores.view(-1, self.num_choices)
        # print(scores)
        # print(labels)

        # loss = self.cross_loss(scores, torch.argmax(labels, axis=1))
        # print(scores)
        # print(torch.max(scores.detach(), labels))
        # loss = self.mse_loss(scores, torch.max(scores.detach(), labels))
        # loss = self.mse_loss(scores, labels.double())

        return (
            # loss + 0.01 * learning_reg,
            loss,
            scores,
            # predictions[:, : self.num_choices],
            is_abstract[:, self.num_choices :],
        )

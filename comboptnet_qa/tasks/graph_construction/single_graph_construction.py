import heapq
import itertools
import math
import time
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import ray
import scipy as sp
import torch
from loguru import logger
from numpy.compat.py3k import is_pathlib_path
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from prefect import Task
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.utils.dummy_pt_objects import prune_layer


class SingleGraphConstruction(Task):
    @staticmethod
    def calc_interoverlap(e1: List, e2: List):
        e1 = set(e1)
        e2 = set(e2)
        score = (
            len(e1.intersection(e2)) / max(len(e1), len(e2))
            if max(len(e1), len(e2)) > 0
            else 0.0
        )
        return score, e1.intersection(e2)

    @staticmethod
    def calc_interoverlap2(e1: List, e2: List):
        e1 = set(e1)
        e2 = set(e2)
        score = len(e1.intersection(e2)) / len(e1) if len(e1) > 0 else 0.0
        return score, e1.intersection(e2)

    @staticmethod
    def calc_interoverlap_question(e1, e2):
        score = (
            len(e1.intersection(e2)) / max(len(e1), len(e2))
            if max(len(e1), len(e2)) > 0
            else 0.0
        )
        return score

    @staticmethod
    def construct_graphs(
        pos,
        input,
        table_store,
        table_store_entities,
        question_enitities,
        num_nodes,
        # question_grounding,
        question_abstract_facts,
        question_embedding,
        grounding_limit,
        entity_mappings,
        table_store_embeddings,
        question_entity_limit=15,
        num_choices=4,
    ):
        NO_OF_NODES = num_nodes + 1
        explanation_graphs = {}
        for q_id, q_exp in tqdm(input.items()):

            # for q_id, question in input.items():
            explanation_graphs[q_id] = {}
            for c_index, choice in enumerate(q_exp["choices"].values()):
                answer = 0
                is_answer = 0
                inference_chains = []
                grounding_nodes = {}
                grounding_abstract_overlap_maps = []
                grounding_lexical_nodes = {}
                q_ovrlp_trms = {}
                choice_mapping = {}
                selected_abstract_nodes = defaultdict(lambda: 0)
                c_id = f"{q_id}|{choice}"
                if c_index >= num_choices:
                    logger.warning(
                        f"{q_id} has more than {num_choices} choice. Ignoring the rest"
                    )
                    continue
                choice_mapping[c_id] = c_index
                if choice == q_exp["answer"]:
                    is_answer = 1
                    answer = c_id

                q_ovrlp_trms = {}

                # for a_id, score in question_abstract_facts[c_id].items():
                for a_id, score in question_abstract_facts[c_id].items():
                    # if len(abstract_nodes[c_id]) > num_nodes - grounding_limit:
                    #     break
                    (
                        overlap_score,
                        overlap_terms,
                    ) = SingleGraphConstruction.calc_interoverlap(
                        question_enitities[c_id], table_store_entities[a_id]
                    )
                    if not overlap_score > 0:
                        continue
                    q_ovrlp_trms[a_id] = overlap_terms

                    # if True:
                    inference_chains.append(
                        (
                            c_id,
                            a_id,
                            # tfidf_score,
                            overlap_score,
                            "abstract",
                        )
                    )
                    a_sem_score = 1 - cosine(
                        question_embedding[c_id], table_store_embeddings[a_id]
                    )
                    selected_abstract_nodes[a_id] = a_sem_score
                    # selected_abstract_nodes[a_id] = max(
                    # score, selected_abstract_nodes[a_id]
                    # )

                selected_abstract_nodes = {
                    id: selected_abstract_nodes[id]
                    for id in heapq.nlargest(
                        num_nodes - grounding_limit,
                        selected_abstract_nodes,
                        key=selected_abstract_nodes.get,
                    )
                }
                # print(selected_abstract_nodes)
                # print(inference_chains)

                if len(selected_abstract_nodes) == 0:
                    logger.error(f"No lexical overalps found for {q_id}. Skipping it")
                # continue

                max_grounding_node_count = grounding_limit
                selected_grounding_node = defaultdict(lambda: 0)
                grounding_nodes = {}
                grounding_lexical_nodes = {}
                for g_id, fact in table_store.items():
                    if fact["type"] == "ABSTRACT":
                        continue

                    g_score = 1
                    if len(grounding_nodes) > max_grounding_node_count - 1:
                        break
                    (
                        overlap_score,
                        overlap_terms,
                    ) = SingleGraphConstruction.calc_interoverlap2(
                        table_store_entities[g_id], question_enitities[c_id]
                    )
                    if g_score > 0 and overlap_score > 0:
                        has_abstract_overlap = False
                        for a_id in selected_abstract_nodes:
                            (
                                a_o_score,
                                a_o_terms,
                            ) = SingleGraphConstruction.calc_interoverlap(
                                table_store_entities[g_id], table_store_entities[a_id]
                            )
                            if (
                                len(
                                    set(["use", "part", "made", "locate"]).intersection(
                                        a_o_terms
                                    )
                                )
                                > 0
                            ):
                                continue
                            if (
                                a_o_score > 0
                                and not overlap_terms == a_o_terms
                                and not len(overlap_terms.intersection(a_o_terms)) > 0
                            ):
                                grounding_abstract_overlap_maps.append((g_id, a_id))
                                selected_grounding_node[g_id] = max(
                                    selected_grounding_node[g_id],
                                    # selected_abstract_nodes[a_id],
                                    overlap_score + a_o_score,
                                )
                                has_abstract_overlap = True
                        if not has_abstract_overlap:
                            continue
                        q_ovrlp_trms[g_id] = overlap_terms
                        grounding_nodes[g_id] = 1
                        grounding_lexical_nodes[g_id] = g_score
                        inference_chains.append(
                            (
                                c_id,
                                g_id,
                                overlap_score,
                                # tfidf_score,
                                "grounding",
                            )
                        )
                selected_grounding_node = {
                    id: selected_grounding_node[id]
                    for id in heapq.nlargest(
                        max_grounding_node_count,
                        selected_grounding_node,
                        key=selected_grounding_node.get,
                    )
                }
                question_abstract_edges = sp.sparse.lil_matrix(
                    (NO_OF_NODES, NO_OF_NODES), dtype=np.float32
                )
                question_grounding_edges = sp.sparse.lil_matrix(
                    (NO_OF_NODES, NO_OF_NODES), dtype=np.float32
                )
                grounding_abstract_edges = sp.sparse.lil_matrix(
                    (NO_OF_NODES, NO_OF_NODES), dtype=np.float32
                )
                abstract_abstract_edges = sp.sparse.lil_matrix(
                    (NO_OF_NODES, NO_OF_NODES), dtype=np.float32
                )
                abstract_abstract_overlap_edges = sp.sparse.lil_matrix(
                    (NO_OF_NODES, NO_OF_NODES), dtype=np.float32
                )
                grounding_grounding_edges = sp.sparse.lil_matrix(
                    (NO_OF_NODES, NO_OF_NODES), dtype=np.float32
                )
                grounding_facts = sp.sparse.lil_matrix(
                    (NO_OF_NODES, NO_OF_NODES), dtype=np.float32
                )
                abstract_facts = sp.sparse.lil_matrix(
                    (NO_OF_NODES, NO_OF_NODES), dtype=np.float32
                )
                # similarity_scores = np.zeros((NO_OF_NODES, NO_OF_NODES), dtype=np.float32)
                similarity_scores = np.zeros((num_nodes), dtype=np.float32)
                is_abstract = np.zeros((NO_OF_NODES), dtype=np.float32)

                node_mapping = {
                    t_id: index + 1
                    for index, t_id in enumerate(
                        {**selected_abstract_nodes, **selected_grounding_node}
                    )
                }

                recall = 0
                # if "explanation" in q_exp:
                #     for t_id in node_mapping:
                #         if (
                #             t_id in q_exp["explanation"]
                #             and q_exp["explanation"][t_id] == "CENTRAL"
                #         ):
                #             if t_id not in abstract_nodes[answer]:
                #                 logger.warning(
                #                     f"{t_id} explanation not connected to {answer}"
                #                 )
                #                 # print(abstract_nodes[answer])
                #                 # print(t_id)

                #                 # print(table_store_entities[t_id], t_id)
                #                 # print(question_enitities[answer])
                #                 continue

                #             recall += 1
                #             similarity_scores[node_mapping[t_id] - num_choices] = 1
                #             # similarity_scores[
                #             #     choice_mapping[answer], node_mapping[t_id]
                #             # ] = 1
                #             # similarity_scores[
                #             #     node_mapping[t_id], choice_mapping[answer]
                #             # ] = 1
                #     if len(q_exp["explanation"]) > 0:
                #         recall = recall / len(q_exp["explanation"])

                assert len(node_mapping) + 1 <= NO_OF_NODES, f"{len(node_mapping)}"

                q_has_abstract_overlap = False
                for chain in inference_chains:
                    c_id, t_id, score, c_type = chain
                    if c_type == "abstract" and t_id in selected_abstract_nodes:
                        question_abstract_edges[0, node_mapping[t_id]] = score
                        is_abstract[node_mapping[t_id]] = 1
                        if score > 0:
                            q_has_abstract_overlap = True
                    elif c_type == "grounding" and t_id in selected_grounding_node:
                        question_grounding_edges[0, node_mapping[t_id]] = score
                # assert q_has_abstract_overlap
                for a1_id in selected_abstract_nodes:
                    for a2_id in selected_abstract_nodes:
                        if a1_id != a2_id:
                            score, _ = SingleGraphConstruction.calc_interoverlap(
                                q_ovrlp_trms[a1_id],
                                q_ovrlp_trms[a2_id],
                            )
                            abstract_abstract_overlap_edges[
                                node_mapping[a1_id], node_mapping[a2_id]
                            ] = score
                            abstract_abstract_edges[
                                node_mapping[a1_id], node_mapping[a2_id]
                            ] = 1 - cosine(
                                table_store_embeddings[a1_id],
                                table_store_embeddings[a2_id],
                            )
                for g_id, a_id in grounding_abstract_overlap_maps:
                    (
                        a_g_score,
                        overlap_terms,
                    ) = SingleGraphConstruction.calc_interoverlap2(
                        q_ovrlp_trms[g_id], q_ovrlp_trms[a_id]
                    )
                    if (
                        g_id in selected_grounding_node
                        and a_id in selected_abstract_nodes
                    ):
                        grounding_abstract_edges[
                            node_mapping[g_id], node_mapping[a_id]
                        ] = a_g_score

                for g1_id in selected_grounding_node:
                    for g2_id in selected_grounding_node:
                        if g1_id != g2_id:
                            (score, _,) = SingleGraphConstruction.calc_interoverlap(
                                table_store_entities[g1_id], table_store_entities[g2_id]
                            )
                            grounding_grounding_edges[
                                node_mapping[g1_id], node_mapping[g2_id]
                            ] = score
                explanation_graphs[q_id][c_id] = {}
                explanation_graphs[q_id][c_id][
                    "question_abstract_edges"
                ] = question_abstract_edges
                explanation_graphs[q_id][c_id][
                    "question_grounding_edges"
                ] = question_grounding_edges
                explanation_graphs[q_id][c_id][
                    "grounding_abstract_edges"
                ] = grounding_abstract_edges
                explanation_graphs[q_id][c_id][
                    "abstract_abstract_edges"
                ] = abstract_abstract_edges
                explanation_graphs[q_id][c_id][
                    "abstract_abstract_overlap_edges"
                ] = abstract_abstract_overlap_edges
                explanation_graphs[q_id][c_id][
                    "grounding_grounding_edges"
                ] = grounding_grounding_edges
                explanation_graphs[q_id][c_id]["grounding_facts"] = grounding_facts
                explanation_graphs[q_id][c_id]["is_abstract"] = is_abstract
                explanation_graphs[q_id][c_id]["abstract_facts"] = abstract_facts
                explanation_graphs[q_id][c_id]["node_mapping"] = {
                    val: key for key, val in node_mapping.items()
                }
                explanation_graphs[q_id][c_id]["choice_mapping"] = {
                    val: key for val, key in choice_mapping.items()
                }
                explanation_graphs[q_id][c_id]["similarity_scores"] = similarity_scores
                # explanation_graphs[q_id][c_id]["recall"] = recall
                explanation_graphs[q_id][c_id]["is_answer"] = is_answer
        return explanation_graphs

    @overrides
    def run(
        self,
        q_dataset,
        table_store,
        table_store_entities,
        question_enitities,
        question_abstract_facts,
        grounding_limit,
        num_nodes,
        table_store_embeddings,
        entity_mappings,
        encoded_hypothesis_text,
        encoded_table_store,
        question_embedding,
        fact_max_len,
        hyp_max_len,
        **kwargs,
    ):

        ray_executor = RayExecutor()
        explanation_graphs = ray_executor.run(
            q_dataset,
            self.construct_graphs,
            dict(
                table_store=table_store,
                table_store_entities=table_store_entities,
                question_enitities=question_enitities,
                question_abstract_facts=question_abstract_facts,
                grounding_limit=grounding_limit,
                num_nodes=num_nodes,
                table_store_embeddings=table_store_embeddings,
                question_embedding=question_embedding,
                entity_mappings=entity_mappings,
            ),
            # batch_count=8,
            **kwargs,
        )

        total_recall = 0
        # for _, val in explanation_graphs.items():
        #     total_recall += val["recall"]

        logger.success(f"Total Skipped: {len(q_dataset)-len(explanation_graphs)}")

        if total_recall > 0:
            logger.success(f"Total Recall: {total_recall/len(explanation_graphs)}")
        else:
            logger.warning("Explanations not provided")

        return TorchDataset(
            explanation_graphs=explanation_graphs,
            q_dataset=q_dataset,
            num_nodes=num_nodes,
            encoded_hypothesis_text=encoded_hypothesis_text["inputs"],
            encoded_table_store=encoded_table_store["inputs"],
            fact_max_len=fact_max_len,
            hyp_max_len=hyp_max_len,
        )


class TorchDataset(Dataset):
    def __init__(
        self,
        explanation_graphs,
        q_dataset,
        num_nodes,
        encoded_hypothesis_text,
        encoded_table_store,
        hyp_max_len,
        fact_max_len,
        num_choices=4,
    ):
        self.explanation_graphs = explanation_graphs
        self.encoded_hypothesis_text = encoded_hypothesis_text
        self.encoded_table_store = encoded_table_store
        self.fact_max_len = fact_max_len
        self.q_dataset = {
            id: val for id, val in q_dataset.items() if id in explanation_graphs
        }
        self.num_choices = num_choices
        self.num_nodes = num_nodes
        self.hyp_max_len = hyp_max_len
        self.key_map = {index: key for index, key in enumerate(self.q_dataset)}

    def build_constraints(
        self,
        is_abstract_fact,
        abstract_limit,
        node_limit,
        outgoing_edges_weigths,
    ):

        NO_OF_NODES = self.num_nodes + 1
        num_variables = NO_OF_NODES * NO_OF_NODES

        # root constraint
        root_constraint_A = torch.zeros((1, num_variables))
        root_constraint_A[0, 0] = -1
        root_constraint_b = torch.Tensor([[1]])

        # incoming constraints
        # edge selection constraint
        edge_selection_A = torch.zeros((NO_OF_NODES * NO_OF_NODES * 3, num_variables))
        edge_selection_b = torch.zeros((NO_OF_NODES * NO_OF_NODES * 3, 1))
        count = 0
        for i in range(0, NO_OF_NODES):
            for j in range(0, NO_OF_NODES):
                if i != j:
                    edge_selection_A[count, i * NO_OF_NODES + i] = -1
                    edge_selection_A[count, i * NO_OF_NODES + j] = 1
                    count += 1
                    edge_selection_A[count, i * NO_OF_NODES + i] = -1
                    # edge_selection_A[count, i] = -1
                    edge_selection_A[count, j * NO_OF_NODES + i] = 1
                    count += 1
                    edge_selection_A[count, i * NO_OF_NODES + i] = 1
                    edge_selection_A[count, j * NO_OF_NODES + j] = 1
                    # edge_selection_A[count, i] = 1
                    # edge_selection_A[count, j] = 1
                    edge_selection_A[count, i * NO_OF_NODES + j] = -1
                    # edge_selection_b[count] = 1
                    edge_selection_b[count] = -1
                    # edge_selection_b[count] = -1
                    # edge_selection_b[count] = -1
                    # edge_selection_b[count] = -1
                    count += 1
        # Abstract fact select
        is_abstract_fact = is_abstract_fact.reshape((NO_OF_NODES, 1))
        is_abstract_fact = torch.Tensor(is_abstract_fact @ is_abstract_fact.T)
        abstract_fact_select_A = (
            torch.diag(torch.ones((NO_OF_NODES))).reshape(NO_OF_NODES * NO_OF_NODES)
            # * is_abstract_fact.reshape(NO_OF_NODES * NO_OF_NODES)
        ).unsqueeze(0)
        abstract_fact_select_b = torch.Tensor([[abstract_limit + 1]]) * -1

        node_select_A = (
            torch.diag(torch.ones((NO_OF_NODES))).reshape(NO_OF_NODES * NO_OF_NODES)
        ).unsqueeze(0)
        node_select_b = torch.Tensor([[node_limit]]) * -1

        A = torch.cat(
            (
                root_constraint_A,
                edge_selection_A,
                abstract_fact_select_A,
                # node_select_A,
            )
        )
        b = torch.cat(
            (
                root_constraint_b,
                edge_selection_b,
                abstract_fact_select_b,
                # node_select_b,
            )
        )
        constraints = torch.cat((A, b), dim=1)

        # # root constraint
        # root_constraint_A = torch.zeros((1, num_variables))
        # root_constraint_A2 = torch.zeros((1, num_variables))
        # for i in range(self.num_choices):
        #     root_constraint_A[0, i] = 1
        #     root_constraint_A2[0, i] = -1
        #     # root_constraint_A[0, 0] = -1
        # root_constraint_b = torch.Tensor([[-1]])
        # root_constraint_b2 = torch.Tensor([[1]])

        # return A, b
        return constraints

    def __getitem__(self, index):
        answer = np.zeros((self.num_choices), dtype=np.float32)

        hypothesis_input_ids = np.zeros(
            (self.num_choices, self.hyp_max_len), dtype=np.long
        )
        hypothesis_attention_mask = np.zeros(
            (self.num_choices, self.hyp_max_len), dtype=np.long
        )
        fact_input_ids = np.zeros(
            (self.num_choices, self.num_nodes, self.fact_max_len), dtype=np.long
        )
        fact_attention_mask = np.zeros(
            (self.num_choices, self.num_nodes, self.fact_max_len), dtype=np.long
        )

        question_abstract_edges = np.zeros(
            (self.num_choices, self.num_nodes + 1, self.num_nodes + 1)
        )
        question_grounding_edges = np.zeros(
            (self.num_choices, self.num_nodes + 1, self.num_nodes + 1)
        )
        abstract_abstract_lexical = np.zeros(
            (self.num_choices, self.num_nodes + 1, self.num_nodes + 1)
        )
        abstract_abstract_similarity = np.zeros(
            (self.num_choices, self.num_nodes + 1, self.num_nodes + 1)
        )
        grounding_grounding_edges = np.zeros(
            (self.num_choices, self.num_nodes + 1, self.num_nodes + 1)
        )
        grounding_abstract_edges = np.zeros(
            (self.num_choices, self.num_nodes + 1, self.num_nodes + 1)
        )

        q_id = self.key_map[index]
        q_exp = self.q_dataset[q_id]

        NO_OF_NODES = self.num_nodes + 1
        constraints = torch.zeros(
            (
                self.num_choices,
                NO_OF_NODES * NO_OF_NODES * 3 + 1 + 1,
                NO_OF_NODES * NO_OF_NODES + 1,
            )
        )
        for c_index, choice in enumerate(q_exp["choices"].values()):
            if c_index >= self.num_choices:
                # logger.warning(f"{q_id} has more than {self.num_choices}")
                continue
            if choice == q_exp["answer"]:
                answer[c_index] = 1

            hypothesis_input_ids[c_index] = self.encoded_hypothesis_text[
                f"{self.key_map[index]}|{choice}"
            ]["input_ids"]
            hypothesis_attention_mask[c_index] = self.encoded_hypothesis_text[
                f"{self.key_map[index]}|{choice}"
            ]["attention_masks"]

        for i in range(self.num_choices - len(q_exp["choices"])):
            hypothesis_input_ids[len(q_exp["choices"]) + i][0] = 101
            hypothesis_input_ids[len(q_exp["choices"]) + i][1] = 102
            hypothesis_attention_mask[len(q_exp["choices"]) + i][0] = 1
            hypothesis_attention_mask[len(q_exp["choices"]) + i][1] = 1

        for c_index, choice in enumerate(q_exp["choices"].values()):
            if c_index >= self.num_choices:
                # logger.warning(f"{q_id} has more than {self.num_choices}")
                continue
            c_id = f"{q_id}|{choice}"

            results = self.explanation_graphs[q_id][c_id]

            similarity_scores = results["similarity_scores"]
            for f_index, f_id in results["node_mapping"].items():
                assert f_index != 0
                fact_input_ids[c_index, f_index - 1] = self.encoded_table_store[f_id][
                    "input_ids"
                ]
                fact_attention_mask[c_index, f_index - 1] = self.encoded_table_store[
                    f_id
                ]["attention_masks"]
            for i in range(self.num_nodes - len(results["node_mapping"])):
                fact_input_ids[c_index, len(results["node_mapping"]) + i][0] = 0
                fact_input_ids[c_index, len(results["node_mapping"]) + i][1] = 1016
                fact_attention_mask[c_index, len(results["node_mapping"]) + i][0] = 1
                fact_attention_mask[c_index, len(results["node_mapping"]) + i][1] = 1

            question_abstract_edges[c_index] = results[
                "question_abstract_edges"
            ].toarray()
            question_grounding_edges[c_index] = results[
                "question_grounding_edges"
            ].toarray()
            abstract_abstract_lexical[c_index] = results[
                "abstract_abstract_overlap_edges"
            ].toarray()
            abstract_abstract_similarity[c_index] = results[
                "abstract_abstract_edges"
            ].toarray()
            grounding_grounding_edges[c_index] = results[
                "grounding_grounding_edges"
            ].toarray()
            grounding_abstract_edges[c_index] = results[
                "grounding_abstract_edges"
            ].toarray()
            is_abstract = results["is_abstract"]

            # print(is_abstract)

            outgoing_edges_weigths = (
                question_abstract_edges
                + question_grounding_edges
                + grounding_abstract_edges
            )
            for i in range(self.num_choices):
                outgoing_edges_weigths[i][i] = 1
            # outgoing_edges_weigths[i][i] = 1

            inter_edges_weights = abstract_abstract_lexical + grounding_grounding_edges

            # start = time.time()
            constraints[c_index] = self.build_constraints(
                is_abstract_fact=is_abstract,
                abstract_limit=2,
                node_limit=5,
                outgoing_edges_weigths=outgoing_edges_weigths,
            )
            # print(constraints.shape)

        # print(time.time() - start)
        return (
            abstract_abstract_lexical,  # 0
            abstract_abstract_similarity,  # 1
            question_abstract_edges,  # 2
            question_grounding_edges,  # 3
            grounding_grounding_edges,  # 4
            grounding_abstract_edges,  # 5
            is_abstract,  # 6
            similarity_scores,  # 7
            hypothesis_input_ids,  # 8
            fact_input_ids,  # 9
            hypothesis_attention_mask,  # 10
            fact_attention_mask,  # 11
            constraints,
            # A,  # 12
            # b,  # 13
            answer,  # 13
            index,  # 14
        )

    def __len__(self):
        # return 100
        return len(self.q_dataset)

    def get_id(self, index):
        return self.key_map[index]

    def get_exp(self, index):
        return self.explanation_graphs[self.key_map[index]]

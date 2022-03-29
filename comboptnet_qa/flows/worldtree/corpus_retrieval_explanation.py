import os
import string
from collections import defaultdict
from uuid import uuid4

import ray
import ujson as json
from dynaconf import settings
from loguru import logger
from nltk.corpus import stopwords
from poly_nlp.tasks.datasets.genericskb import GenericsKBExtractionTask
from poly_nlp.tasks.datasets.worldtree.extraction_tasks import (
    TableStoreExtractionTask,
    WorldTreeExtractionTask,
    WorldTreeVersion,
)
from poly_nlp.tasks.extraction.spacy_processor_task import SpacyProcessorTask
from poly_nlp.tasks.pre_trained.sentence_transformer import (
    SentenceTransformerEncoderTask,
)
from poly_nlp.tasks.preprocessing.question_to_hypothesis import (
    QuestionHypothesisExtractionTask,
)
from poly_nlp.tasks.retrieval.elastic_search import BuildElasticIndex
from poly_nlp.tasks.retrieval.faiss import FaissIndexBuildTask, FaissSearchTask
from poly_nlp.utils.prefect.unhashed_task_runner import UnhasedTaskRunner
from prefect import Flow, Task, tags, task
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from tqdm import tqdm

stop = stopwords.words("english")

ray.init(ignore_reinit_error=True)

WORLDTREE_VERSION = WorldTreeVersion.WorldTree_V2
version = "v2"

checkpoint_dir = settings["checkpoint_dir"]
generics_kb_path = settings["generics_kb"]["best"]
table_store_path = settings[f"worldtree_{version}"]["table_store_path"]
trainded_model = settings["trained_model"]["distilbert_v2"]
easy_tuple_data = settings["tupleinf"]["easy_data"]
challenge_tuple_data = settings["tupleinf"]["challenge_data"]
# trainded_model = "sentence-transformers/msmarco-distilbert-base-v2"


# Setup result handlers

# TASK_NAME = f"corpus_retrieval_explanationlp"
TASK_NAME = f"corpus_retrieval_explanationlp_generic_all"
cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)
logger.add(f"logs/{TASK_NAME}/answer_selection_test.log")

task_checkpoint_dir = f"{checkpoint_dir}/{TASK_NAME}"


# Initate the tasks
# two_step_retriever = TwoStepRetiver(**cache_args)
table_store_extraction_task = TableStoreExtractionTask(**cache_args)
generics_kb_extraction_task = GenericsKBExtractionTask(**cache_args)
transformer_encoder = SentenceTransformerEncoderTask()
# non_cached_transformer_encoder = SentenceTransformerEncoderTask()
index_build_task = FaissIndexBuildTask()
search_task = FaissSearchTask()

# NOTE run export PREFECT__FLOWS__CHECKPOINTING=true


@task
def process_retrieved_data(corpus, retrieved_data):
    corpus_mapping = {index: id for index, id in enumerate(corpus)}
    updated_retrieved_data = {}
    for q_id, retrieve_dict in retrieved_data.items():
        updated_retrieved_data[q_id] = {
            corpus_mapping[index]: val
            for index, val in retrieve_dict.items()
            if index != -1
        }
    return updated_retrieved_data


@task(**cache_args)
def combine_dicts(dict1, dict2):
    return {**dict1, **dict2}


@task
def fact_mapping(corpus):
    return {id: data["fact"] for id, data in corpus.items()}


@task
def filter_facts(corpus, retrieved_facts):
    filtered_corpus = {}
    for _, facts in retrieved_facts.items():
        for id in facts:
            filtered_corpus[id] = corpus[id]
    logger.success(
        f"Original corpus size: {len(corpus)}. Filtered corpus size: {len(filtered_corpus)}"
    )
    return filtered_corpus


@task
def process_tuple_inf():
    corpus = {}
    # for file in [easy_tuple_data]:
    for file in [easy_tuple_data, challenge_tuple_data]:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0 and not line[0].isnumeric() and len(line.split()) < 24:
                    id = str(uuid4())
                    corpus[id] = {}
                    corpus[id]["fact"] = line
    return corpus


@task(**cache_args)
def corpus_retrieval(hypothesis, corpus, k=50):
    # Build the flow
    with Flow("Explanation selection with Tablestore") as flow:
        # corpus = process_tuple_inf()
        # generic_kb = generics_kb_extraction_task(
        #     generics_kb_path,
        #     filter_fn=lambda fact: fact["source"] not in ["ARC"],
        # )
        # corpus = combine_dicts(table_store, generic_kb)
        with tags("table_store"):
            table_store_embeddings = transformer_encoder(
                fact_mapping(corpus),
                model_name_or_path=trainded_model,
            )
        hypothesis_embeddings = transformer_encoder(
            hypothesis,
            model_name_or_path=trainded_model,
        )
        faiss_index = index_build_task(
            table_store_embeddings, "corpus", task_checkpoint_dir, opts={"mips": False}
        )
        retrieved_facts = search_task(faiss_index, hypothesis_embeddings, k=k)
        updated_retrieved_data = process_retrieved_data(corpus, retrieved_facts)
        filtered_corpus = filter_facts(corpus, updated_retrieved_data)

    state = FlowRunner(flow, task_runner_cls=UnhasedTaskRunner).run(
        return_tasks=flow.tasks
    )
    return {
        "retrieved_facts": state.result[updated_retrieved_data]._result.value,
        # "table_store_embeddings": state.result[table_store_embeddings]._result.value,
        "corpus": state.result[filtered_corpus]._result.value,
    }

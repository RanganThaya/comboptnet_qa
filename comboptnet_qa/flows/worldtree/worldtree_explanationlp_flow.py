from collections import defaultdict

import msgpack
import numpy as np
import ray
import ujson as json
from ax import optimize
from bayes_opt_qa.flows.worldtree_qa.utils import table_store_mapping
from bayes_opt_qa.tasks.data_extraction import FactClassificationTask
from bayes_opt_qa.tasks.explanation_construction import (
    EntityExtractionTask,
    GroundingQuestionReplaceTask,
    GroundMappingTask,
)
from bayes_opt_qa.tasks.search.bm25 import BM25FitTask, BM25SearchTask
from bayes_opt_qa.tasks.utils import WorldTreeLemmatizer
from bayes_opt_qa.tasks.utils.conceptnet_extend import ConceptNetExtend
from bayes_opt_qa.tasks.utils.msgpack_serializer import MsgPackSerializer
from comboptnet_qa.models.explanationlp.explanationlp_model import ExplanationLPModel
from comboptnet_qa.tasks.graph_construction.explanationlp_graph_construction import (
    CandidateGraphConstruction,
)
from comboptnet_qa.tasks.trainer.explanationlp_trainer import ExplanationLPTrainer
from dynaconf import settings
from loguru import logger
from poly_nlp.tasks.datasets.arc_dataset.extraction_tasks import ARCDatasetExtraction
from poly_nlp.tasks.datasets.genericskb import GenericsKBExtractionTask
from poly_nlp.tasks.datasets.worldtree.evaluation_tasks import (
    WorldTreeMAPEvaluationTask,
)
from poly_nlp.tasks.datasets.worldtree.extraction_tasks import (
    TableStoreExtractionTask,
    WorldTreeExtractionTask,
    WorldTreeVersion,
)
from poly_nlp.tasks.pre_trained.sentence_transformer import (
    SentenceTransformerEncoderTask,
)
from poly_nlp.tasks.preprocessing.question_to_hypothesis import (
    QuestionHypothesisExtractionTask,
)
from poly_nlp.tasks.preprocessing.transformer_encode_task import TransformerEncodeTask
from poly_nlp.utils.prefect.unhashed_task_runner import UnhasedTaskRunner
from prefect import Flow, tags, task
from prefect.engine import FlowRunner
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from tqdm import tqdm

from .corpus_retrieval_explanation import corpus_retrieval

ray.init(
    ignore_reinit_error=True, object_store_memory=100 * 1024 ** 3, log_to_driver=False
)


WORLDTREE_VERSION = WorldTreeVersion.WorldTree_V2
version = "v2"


# Setup the setting
train_path = settings[f"worldtree_{version}"]["train"]
dev_path = settings[f"worldtree_{version}"]["dev"]
test_path = settings[f"worldtree_{version}"]["test"]

table_store_path = settings[f"worldtree_{version}"]["table_store_path"]
lemmatizer_path = settings[f"worldtree_{version}"]["lemmatizer_path"]
generics_kb_path = settings["generics_kb"]["best"]
arc_easy_path = settings["arc_dataset"]["easy"]
arc_challenge_path = settings["arc_dataset"]["challenge"]
model_dir = settings["model_dir"]
worldtree_path = settings[f"worldtree_{version}"]

standford_model = settings["stanford_model"]


generics_kb_path = settings["generics_kb"]["best"]


checkpoint_dir = settings["checkpoint_dir"]

# Setup result handlers

# TASK_NAME = f"explanationlp_worldtree"
TASK_NAME = f"explanationlp_worldtree_genric_all"
# TASK_NAME = f"explanationlp_test"
cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"{checkpoint_dir}/{TASK_NAME}"),
)
msgpack_cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(
        dir=f"{checkpoint_dir}/{TASK_NAME}", serializer=MsgPackSerializer()
    ),
)
task_checkpoint_dir = f"{checkpoint_dir}/{TASK_NAME}"
# logger.add(f"logs/{TASK_NAME}/answer_selection_arc_challenge_2.log")


@task(**msgpack_cache_args)
def filter_facts(table_store, results):
    filtered_tablestore = {}
    for q_id, res in results.items():
        for t_id in res:
            filtered_tablestore[t_id] = table_store[t_id]
    return filtered_tablestore


@task(**msgpack_cache_args)
def filter_entities(table_store_entites, results):
    filtered_tablestore = {}
    for t_id, _ in table_store_entites.items():
        if t_id in results:
            filtered_tablestore[t_id] = table_store_entites[t_id]
    return filtered_tablestore


@task(**msgpack_cache_args)
def filter_lemmas(table_store, lemmas):
    filtered_lemmas = {}
    for id, text in tqdm(lemmas.items(), "Filtering lemmas"):
        if id in table_store:
            filtered_lemmas[id] = text
    return filtered_lemmas


@task
def read_msgpack(file_name, worldtree, table_store):
    with open(file_name, "rb") as f:
        expl = msgpack.unpackb(f.read(), raw=False)
    updated = {}
    for id, exps in expl.items():
        q_id, _ = id.split("|")
        if q_id in worldtree:
            updated[id] = {
                x: y
                for index, (x, y) in enumerate(exps.items())
                if x in table_store and table_store[x]["type"] == "ABSTRACT"
            }

    return updated


@task
def read_json(file_name, worldtree, table_store, aug_table_store):
    with open(file_name, "r") as f:
        expl = json.load(f)
    updated = {}
    missed_fact = set()
    for q_id, choice_exp in expl.items():
        if q_id in worldtree:
            for choice, exps in choice_exp.items():
                for e in exps:
                    if e["id"] not in table_store:
                        missed_fact.add(e["id"])
                updated[f'{q_id}|{worldtree[q_id]["choices"][choice]}'] = {
                    e["id"]: e["score"]
                    for e in exps
                    if e["id"] in aug_table_store
                    and aug_table_store[e["id"]]["type"] == "ABSTRACT"
                }
    logger.error(f"Missed fact count: {len(missed_fact)}")
    return updated


@task
def combine_dicts(dict1, dict2):
    return {**dict1, **dict2}


@task
def answer_mapping(worldtree):
    return {
        f"{id}|{choice}": f"{question_exp['question']} {choice}"
        for id, question_exp in worldtree.items()
        for choice in question_exp["choices"].values()
    }


@task
def answer_mapping_choice(worldtree):
    return {
        f"{id}|{choice}": {"question": question_exp["question"], "answer": choice}
        for id, question_exp in worldtree.items()
        for choice in question_exp["choices"].values()
    }


@task(**msgpack_cache_args)
def augument_table_store(table_store, fact_classfication):
    updated_table_store = {}
    for id, fact in tqdm(table_store.items(), "Augumenting table store"):
        fact["type"] = fact_classfication[id]["type"]
        if "arg1" in fact_classfication[id]:
            fact["arg1"] = fact_classfication[id]["arg1"]
            fact["arg2"] = fact_classfication[id]["arg2"]
        updated_table_store[id] = fact
    return updated_table_store


@task(**msgpack_cache_args)
def remove_duplicates(table_store, table_store_entities):
    entity_map = {}
    filtered_table_store = {}
    for id, entities in tqdm(table_store_entities.items(), "Processing duplicates"):
        en_str = " ".join(list(sorted(entities)))
        if en_str not in entity_map:
            entity_map[en_str] = id
            filtered_table_store[id] = table_store[id]
    # return table_store
    return filtered_table_store


@task(**msgpack_cache_args)
def worldtree_classification(table_store, is_abstract=True):
    updated_table_store = {}
    for id, fact_info in table_store.items():
        if is_abstract:
            if fact_info["table_name"] not in ["KINDOF", "SYNONYMY"]:
                # if (
                #     fact_info["table_name"] not in ["SYNONYMY"]
                #     and not "kind of" in fact_info["fact"]
                # ):
                updated_table_store[id] = fact_info
                updated_table_store[id]["type"] = "ABSTRACT"
        else:
            if fact_info["table_name"] in ["KINDOF"]:
                # if "kind of" in fact_info["fact"]:
                fact = fact_info["fact"]
                fact = fact.replace(" a kind of ", " ")
                if "a" in fact:
                    fact = fact.replace(" a ", "")
                elif "an" in fact:
                    fact = fact.replace(" an ", "")
                elif "the" in fact:
                    fact = fact.replace("the ", "")
                if fact.startswith("a "):
                    fact = fact.replace("a ", "")
                if fact.startswith("an "):
                    fact = fact.replace("an ", "")

                # print(fact)
                updated_table_store[id] = fact_info
                updated_table_store[id]["fact"] = fact.lstrip().strip()
                updated_table_store[id]["type"] = "GROUNDING"
                updated_table_store[id]["source"] = "Worldtree"

    return updated_table_store


@task(**msgpack_cache_args)
def conceptnet_classification(table_store):
    updated_table_store = {}
    r_table_store = {}
    arg_map = defaultdict(lambda: [])
    for id, fact_info in table_store.items():
        fact = fact_info["fact"]
        if " is " in fact:
            arg1, arg2 = fact.split(" is ")
            # fact = fact.replace(" is ", " is a kind of ")
        elif " are " in fact:
            if len(fact.split(" are ")) != 2:
                continue
            arg1, arg2 = fact.split(" are ")  #
            # fact = fact.replace(" are ", " are kind of ")
        if f"{arg1} {arg2}" in r_table_store:
            continue
        r_table_store[f"{arg1} {arg2}"] = 1
        updated_table_store[id] = fact_info
        updated_table_store[id]["type"] = "GROUNDING"
        updated_table_store[id]["fact"] = fact

    logger.info(f"Original ConceptNet filtered data: {len(table_store)}")
    logger.info(f"ConceptNet filtered data: {len(updated_table_store)}")
    return updated_table_store


@task(**msgpack_cache_args)
def abstract_classification(table_store):
    updated_table_store = {}
    for id, fact_info in table_store.items():
        if fact_info["source"] == "ConceptNet":
            if any([val in fact_info["fact"] for val in ["use", "locate", "part"]]):
                # print(fact_info)
                updated_table_store[id] = fact_info
                updated_table_store[id]["type"] = "ABSTRACT"
        else:
            updated_table_store[id] = fact_info
            updated_table_store[id]["type"] = "ABSTRACT"
    logger.info(f"Abstract fact count {len(updated_table_store)}")
    return updated_table_store


@task(**msgpack_cache_args)
def filter_types(table_store, lemmatized_facts, fact_type):
    return {
        id: lemmatized_facts[id]
        for id, fact in tqdm(table_store.items(), "Processing types")
        if fact["type"] == fact_type
    }


@task(**cache_args)
def entity_map(table_store_entites):
    entity_map = defaultdict(lambda: [])
    for id, entities in table_store_entites.items():
        for et in entities:
            entity_map[et].append(id)
    return entity_map


@task
def extract_key(obj, key):
    return obj[key]


@task(**cache_args)
def extract_corpus(corpus):
    extract_corpus = {}
    for f_id in corpus:
        extract_corpus[f_id] = corpus[f_id]["fact"]
    return extract_corpus


# Initate the tasks
generics_kb_extraction_task = GenericsKBExtractionTask(**msgpack_cache_args)
build_search_index = BM25FitTask(**cache_args)
bm25_search_task = BM25SearchTask(**msgpack_cache_args)
lemmatizer_task = WorldTreeLemmatizer(**cache_args)
entity_extraction = EntityExtractionTask(**msgpack_cache_args)
arc_dataset_extraction = ARCDatasetExtraction(**msgpack_cache_args)
fact_classificaiton_task = FactClassificationTask(**msgpack_cache_args)
grounding_question_replace = GroundingQuestionReplaceTask(**msgpack_cache_args)
ground_mapping_task = GroundMappingTask(**msgpack_cache_args)
table_store_extraction_task = TableStoreExtractionTask(**cache_args)
worldtree_extraction = WorldTreeExtractionTask(**cache_args)
transformer_encoder = SentenceTransformerEncoderTask(**cache_args)
hypothesis_extraction = QuestionHypothesisExtractionTask(
    stanford_model_dir=standford_model, **cache_args
)
conceptnet_extend = ConceptNetExtend(**cache_args)
transformer_encoder_task = TransformerEncodeTask(**cache_args)

# graph_extraction = ExplanationGraphConstruction()
graph_extraction = CandidateGraphConstruction()


# explanation_evaluate = ExplanationEvaluationTask()
eval_task = WorldTreeMAPEvaluationTask()

eval_task_baseline = WorldTreeMAPEvaluationTask()
# trainer = ExplanationLPTrainer(**cache_args)
trainer = ExplanationLPTrainer()


# prediction = prediction_dev

# NOTE run export PREFECT__FLOWS__CHECKPOINTING=true

CHALLENGE = "Challenge"
EASY = "Easy"


# LIMIT = 10
LIMIT = 20
# LIMIT = 20
# LIMIT = 30
# LIMIT = 40
# LIMIT = 40
# LIMIT = 80
# LIMIT = 100
# GROUNDING_LIMIT = LIMIT / 2
# GROUNDING_LIMIT = 30
GROUNDING_LIMIT = 0
# GROUNDING_LIMIT = 0
# GROUNDING_LIMIT = 0
# GROUNDING_LIMIT = 20
# GROUNDING_LIMIT = 60
GROUNDING_ABSTRACT_LIMIT = 30
QUESIONT_ABSTRACT_LIMIT = 50
HYP_MAX_LEN = 64
FACT_MAX_LEN = 32
# TR_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"
TR_MODEL = "sentence-transformers/all-mpnet-base-v2"
# TR_MODEL = "sentence-transformers/roberta-large-nli-stsb-mean-tokens"
S_MODEL = "all-mpnet-base-v2"

# TR_MODEL = f"{trainded_model}/0_Transformer/"


@task
def qa_dataset(split, dataset="worldtree"):
    if dataset == "worldtree":
        with Flow("Test dataset extraction") as flow:
            with tags(split + dataset):
                dataset = worldtree_extraction(worldtree_path[split], WORLDTREE_VERSION)
    else:
        with Flow("Test dataset extraction") as flow:
            with tags(split + dataset):
                dataset = arc_dataset_extraction(arc_challenge_path[split], CHALLENGE)
            # with tags(split + "easy"):
            # arc_easy = arc_dataset_extraction(arc_easy_path[split], CHALLENGE)
            # dataset = combine_dicts(arc_easy, arc_hard)

    state = FlowRunner(flow=flow, task_runner_cls=UnhasedTaskRunner).run(
        return_tasks=flow.tasks
    )
    return state.result[dataset]._result.value


# @task
@task(**cache_args)
def dataset_construction(split="test", dataset="arc", **opts):
    # Build the flow
    with Flow("Question answering with WorldTreeCorpus") as flow:
        with tags(split):
            arc = qa_dataset(split)
            # arc = qa_dataset(split, "arc")
            # arc = arc_dataset_extraction(arc_challenge_path[split], CHALLENGE)
            # arc = worldtree_extraction(worldtree_path[split], WORLDTREE_VERSION)
            lemmatized_question = lemmatizer_task(answer_mapping(arc), lemmatizer_path)
            dev_entities = entity_extraction(lemmatized_question)

        with tags("question_transformer" + split):
            hypothesis = hypothesis_extraction(answer_mapping_choice(arc))
        with tags("question_transformer" + split + S_MODEL):
            question_embedding = transformer_encoder(
                hypothesis, model_name_or_path=S_MODEL
            )
        with tags("question_transformer" + split + TR_MODEL):
            encoded_hypothesis_text = transformer_encoder_task(
                text_input=hypothesis,
                output_path=f"{task_checkpoint_dir}/{split}",
                t_name="hypothesis_transformer",
                transformer_model=TR_MODEL,
                maxlen=HYP_MAX_LEN,
            )

        with tags("table_store grounding"):
            generic_kb_grounding = generics_kb_extraction_task(
                generics_kb_path,
                process_fn=lambda text: text.replace(" isa ", " is a ")
                .replace(" isan ", " is an ")
                .replace(" is a ", " is ")
                .replace(" is an ", " is ")
                .replace("A ", ""),
                filter_fn=lambda fact: fact["source"]
                # not in ["ConceptNet", "WordNet3.0"],
                # not in ["WordNet3.0"],
                in ["ConceptNet"],
            )
            worldtree_grounding = worldtree_classification(
                table_store_extraction_task(table_store_path, original_map=False),
                is_abstract=False,
            )
            grounding_kb = conceptnet_classification(
                conceptnet_extend(
                    combine_dicts(generic_kb_grounding, worldtree_grounding)
                )
            )
        with tags("table_store abstract"):

            # generic_kb_abstract = abstract_classification(
            #     generics_kb_extraction_task(
            #         generics_kb_path,
            #         process_fn=lambda text: text.replace(" isa ", " is a ")
            #         .replace(" isan ", " is an ")
            #         .replace(" is a ", " is ")
            #         .replace(" is an ", " is ")
            #         .replace("A ", ""),
            #         filter_fn=lambda fact: fact["source"]
            #         # not in ["ConceptNet", "WordNet3.0"],
            #         # not in ["WordNet3.0"],
            #         # in ["ARC"],
            #         in ["ARC", "Waterloo", "TupleKB"],
            #     )
            # )

            worldtree_table_store = worldtree_classification(
                table_store_extraction_task(table_store_path, original_map=True)
            )
            # abstract_table_store = combine_dicts(
            #     generic_kb_abstract, worldtree_table_store
            # )
            # aug_table_store = combine_dicts(grounding_kb, abstract_table_store)
            aug_table_store = combine_dicts(grounding_kb, worldtree_table_store)
            lemmatized_facts = lemmatizer_task(
                table_store_mapping(aug_table_store), lemmatizer_path
            )
            table_store_entites = entity_extraction(lemmatized_facts)
            #     aug_table_store = remove_duplicates(aug_table_store, table_store_entites)
            #     table_store_entites = filter_entities(table_store_entites, aug_table_store)
            lemmatized_facts = filter_lemmas(aug_table_store, lemmatized_facts)
        with tags("table_store abstract" + TR_MODEL):
            encoded_table_store = transformer_encoder_task(
                text_input=extract_corpus(aug_table_store),
                output_path=f"{task_checkpoint_dir}",
                t_name="table_store_transformer",
                transformer_model=TR_MODEL,
                maxlen=FACT_MAX_LEN,
            )
        with tags("GROUNDING_FACTS" + split):
            grounding_facts = filter_types(
                aug_table_store, lemmatized_facts, "GROUNDING"
            )
        # with tags("ABSTRACT_FACTS" + split):
        #     abstract_facts = filter_types(aug_table_store, lemmatized_facts, "ABSTRACT")
        with tags("abstract_retrieve" + split):
            abstract_results = corpus_retrieval(
                hypothesis,
                # abstract_table_store,
                worldtree_table_store,
                k=LIMIT
                # hypothesis, worldtree_table_store, k=LIMIT
            )
        with tags("grounding_retrieve" + split):
            grounding_retriever = build_search_index(grounding_facts)
            grounding_results = bm25_search_task(
                lemmatized_question,
                grounding_retriever,
                limit=10,
            )
        with tags("table_store_transformer" + split + S_MODEL):
            table_store_embeddings = transformer_encoder(
                table_store_mapping(
                    # aug_table_store, grounding_results, abstract_results
                    aug_table_store,
                    abstract_results,
                ),
                model_name_or_path=S_MODEL,
            )

        with tags(
            str(LIMIT)
            + " "
            + str(int(opts.get("grounding_limit", GROUNDING_LIMIT)))
            + split
        ):
            dataset = graph_extraction(
                q_dataset=arc,
                lemmatized_questions=lemmatized_question,
                table_store=aug_table_store,
                table_store_entities=table_store_entites,
                entity_mappings=entity_map(table_store_entites),
                question_enitities=dev_entities,
                question_abstract_facts=extract_key(
                    abstract_results, "retrieved_facts"
                ),
                question_grounding=grounding_results,
                num_nodes=int(opts.get("num_nodes", LIMIT)),
                grounding_limit=int(opts.get("grounding_limit", GROUNDING_LIMIT)),
                table_store_embeddings=table_store_embeddings,
                opts=opts,
                encoded_hypothesis_text=encoded_hypothesis_text,
                encoded_table_store=encoded_table_store,
                fact_max_len=FACT_MAX_LEN,
                question_embedding=question_embedding,
                hyp_max_len=HYP_MAX_LEN,
                # split=split,
                # question_corpus=arc,
            )

    state = FlowRunner(flow=flow, task_runner_cls=UnhasedTaskRunner).run(
        return_tasks=flow.tasks
    )
    return {
        "dataset": state.result[dataset]._result.value,
        "q_dataset": state.result[arc]._result.value,
        "table_store": state.result[aug_table_store]._result.value,
    }


@task
def init_model(disable_transformer=False, only_answer=False, opts={}):
    model = ExplanationLPModel(
        transformer_model=TR_MODEL,
        num_nodes=LIMIT,
        hyp_max_len=HYP_MAX_LEN,
        fact_max_len=FACT_MAX_LEN,
        disable_transformer=disable_transformer,
        only_answer=only_answer,
        opts=opts,
    )

    return model


@task
def precison_evaluate(dataset, eval_results, k=1):
    node_preds = eval_results[2]
    answer_preds = eval_results[1]
    torch_dataset = dataset["dataset"]
    q_dataset = dataset["q_dataset"]
    table_store = dataset["table_store"]

    pred_output = {}
    # sorted_node_preds = np.flip(np.argsort(node_preds, axis=-1))[:, :2]
    mean_precision, count, total_recalled = 0, 0, 0
    with open(f"pred_full_{k}.text", "w") as f:

        for index, exp_scores in enumerate(node_preds):
            exp_indexes = np.flip(np.argsort(exp_scores))[:k]
            count += 1
            q_id = torch_dataset.get_id(index)
            node_mapping = torch_dataset.explanation_graphs[q_id]["node_mapping"]
            exps = [node_mapping[i + 4] for i in exp_indexes if i < len(node_mapping)]

            for c_index, choice in enumerate(q_dataset[q_id]["choices"].values()):
                if c_index == answer_preds[index]:
                    pred_output[q_id] = {
                        "question": q_dataset[q_id]["question"],
                        "id": q_id,
                        "predicted_answer": choice,
                        "correct_answer": q_dataset[q_id]["answer"],
                        "sentences": [table_store[e]["fact"] for e in exps],
                        "explanations": [
                            table_store[e]["fact"]
                            for e in q_dataset[q_id]["explanation"]
                            if e in table_store
                        ],
                        # "explanations_id": list(q_dataset[q_id]["explanation"].keys()),
                        # "exps": exps,
                    }

            exps = list(set(exps))
            recalled = 0
            for exp in exps:
                exp = exp.split("|")[0]
                if (
                    exp
                    in q_dataset[q_id]["explanation"]
                    # and q_dataset[q_id]["explanation"][exp] == "CENTRAL"
                ):
                    recalled += 1
            total_recalled += recalled
            mean_precision += recalled / k
            pred_output[q_id]["precision"] = recalled / k
            json.dump(pred_output[q_id], f)
            f.write("\n")

    logger.success(f"Total recalled: {total_recalled}")
    logger.success(f"Precision@K: {mean_precision/count}")
    logger.success(f"Micro precision@K: {total_recalled/(k*count)}")


def explanation_evaluate(
    lambda_val=2,
    # disable_transformer=False, only_answer=False, mode="train", opts={}, lambda_val=2
):
    with Flow("Question answering with WorldTreeCorpus") as flow:
        with tags("train" + "worldtree"):
            train_dataset = dataset_construction(split="train", dataset="worldtree")
        # with tags("test"):
        #     test_dataset = explanation_construction(split="test")
        with tags("test" + "worldtree"):
            dev_dataset = dataset_construction(split="test", dataset="worldtree")
        # with tags("dev" + "worldtree"):
        #     test_dataset = dataset_construction(split="dev", dataset="worldtree")
        # # with tags("train" + "arc"):
        # #     train_dataset_arc = dataset_construction(split="train", dataset="arc")
        # # with tags("test" + "arc"):
        # #     dev_dataset_arc = dataset_construction(split="test", dataset="arc")
        eval_results = trainer(
            train_dataset=extract_key(train_dataset, "dataset"),
            dev_dataset=extract_key(dev_dataset, "dataset"),
            # dev_dataset=extract_key(dev_dataset, "dataset"),
            # test_dataset=extract_key(test_dataset, "dataset"),
            # load_task_name="alpha_optimized_explanation",
            # task_name="full_optimized_explanation_arc_tuple",
            # load_task_name="full_optimized_explanation",
            model=init_model(
                opts={**lambda_val}
                # disable_transformer, only_answer, {"lambda_val": lambda_val}
            ),
            task_name="ExplanationLP_Full",
            # task_name="ExplanationLP_Answer",
            # task_name="ApproxExplanationLP",
            output_dir=model_dir,
            mode="train",
            # mode=mode,
            eval_fn=None,
        )
    state = FlowRunner(flow=flow, task_runner_cls=UnhasedTaskRunner).run(
        return_tasks=flow.tasks
    )
    return state.result[eval_results]._result.value[0]
    # return {
    # "dataset": state.result[best_score]._result.value,
    # }


# score = explanation_evaluate(False, False, "train")


optimize(
    parameters=[
        {
            "name": "lambda_val",
            "type": "range",
            "bounds": [1, 100],
        },
    ],
    # Booth function
    evaluation_function=explanation_evaluate,
    minimize=False,
)

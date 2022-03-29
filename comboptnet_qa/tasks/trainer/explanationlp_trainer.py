import os
import random

import numpy as np
import torch
from loguru import logger
from overrides import overrides
from prefect import Task
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

BERT_MODEL = "bert-base-uncased"


class ExplanationLPTrainer(Task):
    def __init__(self, **kwargs):
        super(ExplanationLPTrainer, self).__init__(**kwargs)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 15)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 2)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 4)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 2)
        self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 8)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 2)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 4)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 8)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 8)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 6)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 6)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 6)
        # self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 6)
        self.cuda = kwargs.get("cuda", True)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        # self.num_train_epochs = kwargs.get("num_train_epochs", 14)
        # self.num_train_epochs = kwargs.get("num_train_epochs", 50)
        self.num_train_epochs = kwargs.get("num_train_epochs", 8)
        self.learning_rate = kwargs.get("learning_rate", 1e-5)
        # self.learning_rate = kwargs.get("learning_rate", 1e-5)
        # self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.weight_decay = kwargs.get("weight_decay", 1e-5)
        self.adam_epsilon = kwargs.get("adam_epsilon", 1e-8)
        self.warmup_steps = kwargs.get("warmup_steps", 30)
        # self.warmup_steps = kwargs.get("warmup_steps", 10)
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.logging_steps = kwargs.get("logging_steps", 1)
        self.args = kwargs

    def set_seed(self, n_gpu, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    @overrides
    def run(
        self,
        train_dataset,
        dev_dataset,
        task_name,
        output_dir,
        model,
        mode="train",
        eval_fn=None,
        save_optimizer=True,
        load_task_name=None,
        test_dataset=None,
    ):
        torch.cuda.empty_cache()
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count() if self.cuda else 0
        # n_gpu = 1
        self.logger.info(f"GPUs used {n_gpu}")
        train_batch_size = self.per_gpu_batch_size * max(1, n_gpu)
        test_batch_size = train_batch_size

        # train_dataset, dev_dataset, test_dataset = datasets
        # train_dataset = train_dataset
        # dev_dataset = test_dataset
        # test_dataset = test_dataset

        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=test_batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False
        )

        self.set_seed(n_gpu)
        outputs = {}
        if load_task_name is not None:
            logger.info(f"Loading from {output_dir}/{load_task_name}")
            model.load_state_dict(
                torch.load(os.path.join(output_dir, load_task_name, "model.pt"))
            )
        if mode == "train":
            logger.info("Running train mode")
            model = model.to(device)
            model = model.double()
            if n_gpu > 1 and self.cuda:
                model = torch.nn.DataParallel(model)
            epoch_results, best_score = self.train(
                model,
                train_dataloader,
                dev_dataloader,
                dev_dataset,
                device,
                n_gpu,
                eval_fn,
                f"{output_dir}/{task_name}",
                save_optimizer,
                root_dir=output_dir,
                load_task_name=load_task_name,
            )
            outputs["epoch_results "] = epoch_results
        logger.info("Running evalutaion mode")
        logger.info(f"Loading from {output_dir}/{task_name}")
        # model.load_state_dict(
        #     torch.load(os.path.join(f"{output_dir}/{task_name}", "model.pt"))
        # )
        # model.to(device)
        # if mode == "eval":
        #     best_score, answer_preds, node_preds = self.eval(
        #         model,
        #         test_dataloader,
        #         test_dataset,
        #         device,
        #         n_gpu,
        #         eval_fn,
        #         mode="dev",
        #     )
        #     return best_score, answer_preds, node_preds
        # if test_dataset is not None:
        #     test_data_loader = DataLoader(
        #         test_dataset, batch_size=test_batch_size, shuffle=False
        #     )
        #     score = self.eval(
        #         model,
        #         test_data_loader,
        #         test_dataset,
        #         device,
        #         n_gpu,
        #         eval_fn,
        #         mode="test",
        #     )
        return best_score, None, None

    def train(
        self,
        model,
        train_dataloader,
        dev_dataloader,
        dev_dataset,
        device,
        n_gpu,
        eval_fn,
        output_dir,
        save_optimizer,
        root_dir,
        load_task_name=None,
    ):
        results = {}
        best_score = 0.0
        t_total = (
            len(train_dataloader)
            // self.gradient_accumulation_steps
            * self.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        low_lr = ["clamp_param"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in low_lr)
                ],
                "weight_decay": self.weight_decay,
                # "lr": 1e-4,
                "lr": 5e-5,
                # "lr": 1e-5,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                # "lr": 1e-4,
                "lr": 5e-5,
                # "lr": 1e-6,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in low_lr)
                ],
                "weight_decay": 0.0,
                # "lr": 1,
                # "lr": 1e-1,
                "lr": 1e-2,
                # "lr": 1e-4,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total,
        )
        # if load_task_name is not None:
        #     logger.info(
        #         f"Loading optimizer and scheduler from {output_dir}/{load_task_name}"
        #     )
        #     optimizer.load_state_dict(
        #         torch.load(os.path.join(root_dir, load_task_name, "optimizer.pt"))
        #     )
        #     scheduler.load_state_dict(
        #         torch.load(os.path.join(root_dir, load_task_name, "scheduler.pt"))
        #     )

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained,
            int(self.num_train_epochs),
            desc="Epoch",
        )
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "hypothesis_input_ids": batch[8],
                    "fact_input_ids": batch[9],
                    "hypothesis_attention_mask": batch[10],
                    "fact_attention_mask": batch[11],
                    "constraints": batch[12],
                    # "A": batch[12],
                    # "b": batch[13],
                    "labels": batch[13],
                    "abstract_abstract_lexical": batch[0],  # 0
                    "abstract_abstract_similarity": batch[1],  # 1
                    "question_abstract_edges": batch[2],  # 2
                    "question_grounding_edges": batch[3],  # 3
                    "grounding_grounding_edges": batch[4],  # 4
                    "grounding_abstract_edges": batch[5],  # 5
                    "is_abstract": batch[6],  # 7
                    "similarity_scores": batch[7],  # 8
                    "epoch": epoch,
                }
                outputs = model(**inputs)
                loss = outputs[
                    0
                ]  # model outputs are always tuple in transformers (see doc)

                if n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        loss_scalar = (tr_loss - logging_loss) / self.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        epoch_iterator.set_description(
                            f"Loss :{loss_scalar} LR: {learning_rate_scalar}"
                        )
                        logging_loss = tr_loss
            score = self.eval(
                model,
                dev_dataloader,
                dev_dataset,
                device,
                n_gpu,
                eval_fn,
                mode="dev",
            )[0]
            results[epoch] = score
            with torch.no_grad():
                if score > best_score:
                    logger.success(f"Storing the new model with score: {score}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # model_to_save = (
                    # model.module if hasattr(model, "module") else model
                    # )  # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    torch.save(
                        model_to_save.state_dict(),
                        os.path.join(output_dir, "model.pt"),
                    )

                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    logger.info(f"Saving model checkpoint to {output_dir}")
                    if save_optimizer:
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir
                        )
                    best_score = score

        return results, best_score

    def eval(self, model, dataloader, dataset, device, n_gpu, eval_fn, mode):
        if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel) and self.cuda:
            model = torch.nn.DataParallel(model)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds, node_preds = None, None
        indexes = None
        out_label_ids = None
        for batch in tqdm(dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "hypothesis_input_ids": batch[8],
                    "fact_input_ids": batch[9],
                    "hypothesis_attention_mask": batch[10],
                    "fact_attention_mask": batch[11],
                    "constraints": batch[12],
                    # "A": batch[12],
                    # "b": batch[13],
                    # "labels": batch[14],
                    "labels": batch[13],
                    "abstract_abstract_lexical": batch[0],  # 0
                    "abstract_abstract_similarity": batch[1],  # 1
                    "question_abstract_edges": batch[2],  # 2
                    "question_grounding_edges": batch[3],  # 3
                    "grounding_grounding_edges": batch[4],  # 4
                    "grounding_abstract_edges": batch[5],  # 5
                    "is_abstract": batch[6],  # 7
                    "similarity_scores": batch[7],  # 8
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits, node_logits = outputs[:3]

                # eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                node_preds = node_logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                # indexes = batch[9].detach().cpu().numpy()
            else:
                preds = np.vstack((preds, logits.detach().cpu().numpy()))
                node_preds = np.vstack((node_preds, node_logits.detach().cpu().numpy()))
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )
                # indexes = np.append(indexes, batch[9].detach().cpu().numpy(), axis=0)

        # eval_loss = eval_loss / nb_eval_steps

        score = None
        # if eval_fn is not None:
        #     score = eval_fn(
        #         dataset=dataset,
        #         indexes=indexes,
        #         preds=preds,
        #         out_label_ids=out_label_ids,
        #         mode=mode,
        #     )
        preds = np.argmax(preds, axis=1)
        score = (preds == np.argmax(out_label_ids, axis=1)).mean()

        logger.info(f"Score:{score}")
        return score, preds, node_preds

import os

try:
    from transformers import Seq2SeqTrainer
except ImportError:
    Seq2SeqTrainer = object
import torch.nn as nn

class TrainerForAuto(Seq2SeqTrainer):
    def __init__(self,teacher=None,alpha_ce=0.5,alpha_task=0.5,temperature=2.0,**params):
        super().__init__(**params)
        self.teacher = teacher
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.alpha_ce = alpha_ce
        self.alpha_task = alpha_task
        self.temperature = temperature

    def compute_loss(self, model, inputs,return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss_task = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss_task = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.teacher is not None:
            self.teacher.eval()
            stu_logits = outputs["logits"]
            output_tea = self.teacher(**inputs)
            tea_logits = output_tea["logits"]
            loss_ce = self.loss_fn(
                nn.functional.log_softmax(stu_logits / self.temperature, dim=-1),
                nn.functional.softmax(tea_logits / self.temperature, dim=-1),
            )*(self.temperature ** 2)
            loss = self.alpha_ce * loss_ce + self.alpha_task * loss_task
        else:
            loss = loss_task

        return (loss, outputs) if return_outputs else loss

    def predict(
        self,
        test_dataset,
        ignore_keys=None,
        metric_key_prefix=None,
        max_length=None,
        num_beams=None,
    ):
        if getattr(self, "_is_seq2seq", None):
            return super().predict(
                test_dataset,
                ignore_keys,
                metric_key_prefix,
                max_length,
                num_beams,
            )
        else:
            return super(Seq2SeqTrainer, self).predict(
                test_dataset, ignore_keys, metric_key_prefix
            )

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):
        if getattr(self, "_is_seq2seq", None):
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
        else:
            return super(Seq2SeqTrainer, self).prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )



    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Overriding transformers.Trainer.evaluate by saving metrics and checkpoint path."""
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        ckpt_dir = os.path.join(
            self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # TODO: if your task is seq2seq (i.e., SUMMARIZATION), uncomment the code below (add indentation before metrics = eval_dataset...

        if getattr(self, "_is_seq2seq", None):
            metrics = eval_dataset and super().evaluate(
                eval_dataset,
                ignore_keys,
                metric_key_prefix,
                max_length=self.args.generation_max_length,
                num_beams=self.args.generation_num_beams,
            )
        else:
            metrics = eval_dataset and super(Seq2SeqTrainer, self).evaluate(
                eval_dataset,
                ignore_keys,
                metric_key_prefix,
            )
        if metrics:
            for key in list(metrics.keys()):
                if key.startswith("eval_"):
                    metrics[key[5:]] = metrics.pop(key)
        if hasattr(self, "ckpt_to_global_step"):
            self.ckpt_to_global_step[ckpt_dir] = self.state.global_step
            if metrics:
                self.ckpt_to_metric[ckpt_dir] = metrics
        else:
            self.ckpt_to_global_step = {ckpt_dir: self.state.global_step}
            self.ckpt_to_metric = {ckpt_dir: metrics} if metrics else {}


# TODO: if your task is SUMMARIZATION, you need a different
#  class Seq2SeqTrainerForAuto, uncomment the code below
#  Note: I have implemented it here,
#  but I don't know whether it's correct, you need to debug
#  Seq2SeqTrainerForAuto to make sure it's correct


# class Seq2SeqTrainerForAuto(TrainerForAuto):
#     def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
#         """Overriding transformers.Trainer.evaluate by saving metrics and checkpoint path"""
#         self._is_seq2seq = True
#         TrainerForAuto.evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
#         # super(TrainerForAuto, self).evaluate(
#         #     eval_dataset, ignore_keys, metric_key_prefix
#         # )


# TODO: if your task is QUESTIONANSWERING, uncomment the code below
#  by adapting the code in https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py#L28


# class QATrainerForAuto(TrainerForAuto):
#     pass
# TODO: if your task is QUESTIONANSWERING, do the post processing here

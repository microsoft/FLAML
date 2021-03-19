import copy
import os

import transformers
from ray import tune
import torch
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class TrainerForAutoHF(transformers.Trainer):
    """
        Overriding transformers.Trainer.

        Args:
            model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
    """

    def get_optimizers(
            self, num_training_steps
    ):
        self.current_optimizer, self.current_scheduler = super().get_optimizers(num_training_steps)
        return (self.current_optimizer, self.current_scheduler)

    def evaluate(self,
                 eval_dataset= None):
        """
                Overriding transformers.Trainer.evaluate by saving state with save_state

                Args:
                    eval_dataset:
                        the dataset to be evaluated
            """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(
            eval_dataloader, description="Evaluation")
        self.log(output.metrics)

        self.save_state()

        output_metrics = copy.deepcopy(output.metrics)
        for key in output.metrics.keys():
            if key.startswith("eval_"):
                output_metrics[key[5:]] = output_metrics[key]

        tune.report(**output_metrics)

        return output_metrics

    def save_state(self):
        """
                Overriding transformers.Trainer.save_state. It is only through saving
                the states can best_trial.get_best_checkpoint return a non-empty value.
        """
        with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
            self.args.output_dir = checkpoint_dir
            # This is the directory name that Huggingface requires.
            output_dir = os.path.join(
                self.args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self.save_model(output_dir)
            torch.save(self.optimizer.state_dict(),
                       os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(),
                       os.path.join(output_dir, "scheduler.pt"))

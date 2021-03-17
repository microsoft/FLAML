import copy
import os

import transformers
from ray import tune
import torch
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class TrainerForAutoHF(transformers.Trainer):
    def get_optimizers(
            self, num_training_steps
    ):
        self.current_optimizer, self.current_scheduler = super(
        ).get_optimizers(num_training_steps)
        return (self.current_optimizer, self.current_scheduler)

    def evaluate(self,
                 eval_dataset= None):
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
        with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
            self.args.output_dir = checkpoint_dir
            # This is the directory name that Huggingface requires.
            output_dir = os.path.join(
                self.args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self.save_model(output_dir)
            #if self.is_world_master():
            torch.save(self.optimizer.state_dict(),
                       os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(),
                       os.path.join(output_dir, "scheduler.pt"))
import numpy as np
import torch

from transformers.trainer_utils import set_seed

set_seed(42)
torch.manual_seed(42)

def get_batch_jacobian(model, inputs, trainer):
    from transformers import AutoModelForSequenceClassification
    from torchviz import make_dot

    loss, output = trainer.compute_loss(model, inputs, return_outputs=True)
    #output = model(x['input_ids'])
    loss.backward(retain_graph=True)

    hidden_states = output.hidden_states

    X = hidden_states[0]
    X.requires_grad_(True)
    y = output['loss']

    jacob = torch.autograd.grad(
        y, X,
        retain_graph=False,
        create_graph=False,)
        # grad_outputs=torch.ones_like(y))

    return jacob[0].detach()

def eval_score(jacob):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))

def compute_jacob_cov(net, inputs, trainer):
    device = list(inputs.values())[0].device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    jacobs = get_batch_jacobian(net, inputs, trainer)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        jc = eval_score(jacobs)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc

# def load_model(config=None):
#     from flaml.nlp import AutoTransformers
#     from transformers import AutoConfig, AutoModelForSequenceClassification
#     training_args_config, per_model_config = AutoTransformers._separate_config(config)
#     model_config = AutoConfig.from_pretrained(
#         pretrained_model_name_or_path="google/electra-base-discriminator",
#         output_hidden_states=True,
#         **per_model_config)
#     model_config.output_hidden_states = True
#     model = AutoModelForSequenceClassification.from_pretrained(
#         pretrained_model_name_or_path='bert-base-uncased',
#         config=model_config)
#     return model

def get_trainer(config, autohf):
    from flaml.nlp import AutoTransformers
    from transformers import TrainingArguments
    training_args_config, per_model_config = AutoTransformers._separate_config(config)
    this_model = autohf._load_model(per_model_config=per_model_config)

    def model_init():
        return autohf._load_model(per_model_config=per_model_config)

    set_seed(42)
    torch.manual_seed(42)

    autohf._resources_per_trial = {"gpu": 1, "cpu": 1}
    autohf.ckpt_per_epoch = 1

    ckpt_freq = autohf._compute_checkpoint_freq(
        num_train_epochs=config["num_train_epochs"],
        batch_size=config["per_device_train_batch_size"])

    from transformers import IntervalStrategy
    from flaml.nlp.huggingface.trainer import TrainerForAutoTransformers
    training_args = TrainingArguments(
        output_dir="./data/",
        do_train=True,
        do_eval=True,
        per_device_eval_batch_size=32,
        eval_steps=ckpt_freq,
        evaluation_strategy=IntervalStrategy.STEPS,
        save_steps=ckpt_freq,
        save_total_limit=0,
        fp16=True,
        **training_args_config,
    )

    trainer = TrainerForAutoTransformers(
        model=this_model,
        args=training_args,
        model_init=model_init,
        train_dataset=autohf.train_dataset,
        eval_dataset=autohf.eval_dataset,
        tokenizer=autohf._tokenizer,
        compute_metrics=autohf._compute_metrics_by_dataset_name,
    )

    return this_model, trainer

def get_onebatch_proxy_score(autohf,
                             jobid_config=None,
                             hp_config=None,
                             console_args=None):

    from run_autohf import get_preparedata_setting
    preparedata_setting = get_preparedata_setting(console_args, jobid_config)
    autohf.prepare_data(**preparedata_setting)
    this_model, trainer = get_trainer(hp_config, autohf)

    trainloader = trainer.get_train_dataloader()

    inputs = get_some_data(trainloader,
                           device=torch.device("cuda:0"
                           if torch.cuda.is_available() else "cpu"))

    metric = compute_jacob_cov(this_model, inputs, trainer)
    #metric = compute_synflow_per_weight(this_model, inputs, trainer)

    return metric

def get_some_data(train_dataloader, device):
    dataloader_iter = iter(train_dataloader)
    outputs = next(dataloader_iter)
    for key in list(outputs.keys()):
        outputs[key] = outputs[key].to(device)
    return outputs

def get_layer_metric_array(net, metric):
    import torch.nn as nn
    metric_array = []

    for layer in net.modules():
        #if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        try:
            metric_layer = metric(layer)
            metric_array.append(metric_layer)
        except AttributeError:
            pass

    return metric_array

def compute_synflow_per_weight(net, inputs, trainer):
    device = list(inputs.values())[0].device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    loss, output = trainer.compute_loss(net,
                            inputs,
                            return_outputs=True)
    loss.backward(retain_graph=True)

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    def sum_arr(arr):
        sum = 0.
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    grads_abs = get_layer_metric_array(net, synflow)

    # apply signs of all params
    nonlinearize(net, signs)

    return sum_arr(grads_abs)
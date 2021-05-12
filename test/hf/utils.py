import json, os

def get_wandb_azure_key(key_path):
    key_json = json.load(open(os.path.join(key_path, "key.json"), "r"))
    wandb_key = key_json["wandb_key"]
    azure_key = key_json["azure_key"]
    return wandb_key, azure_key
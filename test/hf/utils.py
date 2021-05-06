import json

def get_wandb_azure_key():
    key_json = json.load(open("key.json", "r"))
    wandb_key = key_json["wandb_key"]
    azure_key = key_json["azure_key"]
    return wandb_key, azure_key
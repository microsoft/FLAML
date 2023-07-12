import os
from pseudo_main import pseudo_main
from flaml import oai
import json
import openai


def main():
    pseudo_main(config_list)


if __name__ == "__main__":
    config_list = None
    try:
        openai.api_key = open("key_openai.txt").read().strip()
    except Exception:
        from azure.identity import DefaultAzureCredential

        SCOPE = "https://ml.azure.com"
        credential = DefaultAzureCredential()
        token = credential.get_token(SCOPE).token
        headers = {
            "azureml-model-deployment": "gpt4",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **json.load(open("headers.json")),
        }
        config_list = [
            {
                "api_key": open("key.txt").read().strip(),
                "api_type": "open_ai",
                "api_base": "https://api.openai.com/v1",
            },
            {
                "api_key": open("key_flaml.txt").read().strip(),
                "api_type": "azure",
                "api_base": open("base_flaml.txt").read().strip(),
                "api_version": "2023-03-15-preview",
            },
            {
                "api_key": open("key_aoai.txt").read().strip(),
                "api_type": "azure",
                "api_base": open("base_aoai.txt").read().strip(),
                "api_version": "2023-03-15-preview",
            },
            # {
            #     "api_key": open("key_gcr.txt").read().strip(),
            #     "api_type": "azure",
            #     "api_base": open("base_gcr.txt").read().strip(),
            #     "api_version": "2023-03-15-preview",
            # },
            # {
            #     "headers": headers,
            #     "api_base": open("base_azure.txt").read().strip(),
            # },
        ]

    try:
        os.environ["WOLFRAM_ALPHA_APPID"] = open("wolfram.txt").read().strip()
    except Exception:
        print("Warning: Wolfram Alpha API key not found.")
        pass
    oai.retry_timeout = 3600
    main()

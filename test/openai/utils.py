import os


def config_list_gpt4_gpt35():
    return [
        {
            "model": "gpt-4",
        },
        {
            "model": "gpt-4",
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "api_type": "azure",
            "api_base": os.environ.get("AZURE_OPENAI_API_BASE"),
            "api_version": "2023-03-15-preview",
        },
        {
            "model": "gpt-3.5-turbo",
            "api_type": "open_ai",
            "api_base": "https://api.openai.com/v1",
        },
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "api_type": "azure",
            "api_base": os.environ.get("AZURE_OPENAI_API_BASE"),
            "api_version": "2023-03-15-preview",
        },
    ]


def config_list_openai_aoai():
    return [
        {},
        {
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "api_type": "azure",
            "api_base": os.environ.get("AZURE_OPENAI_API_BASE"),
            "api_version": "2023-03-15-preview",
        },
    ]

from flaml import oai


def test_adv_gen():
    try:
        import openai
    except ImportError:
        return

    # input_examples = [
    #     {}, {}
    # ]

    # adv_examples, metric_change = oai.Completion.generate_adversarial_examples(data=input_example, metric, mode, eval_func, num_examples)

    # adv_examples are like ...

    # metric is changed from ... to ...


if __name__ == "__main__":
    import openai

    openai.api_key_path = "test/openai/key.txt"
    # if you use Azure OpenAI, comment the above line and uncomment the following lines
    # openai.api_type = "azure"
    # openai.api_base = "https://<your_endpoint>.openai.azure.com/"
    # openai.api_version = "2023-03-15-preview"  # change if necessary
    # openai.api_key = "<your_api_key>"
    test_adv_gen()

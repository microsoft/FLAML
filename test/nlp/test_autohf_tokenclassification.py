import sys
import pytest
import requests


@pytest.mark.skipif(sys.platform == "darwin", reason="do not run on mac os")
def test_tokenclassification():
    from flaml import AutoML
    import pandas as pd

    train_data = {
        "chunk_tags": [
            [11, 21, 11, 12, 21, 22, 11, 12, 0],
            [11, 12],
            [11, 12],
            [
                11,
                12,
                12,
                21,
                13,
                11,
                11,
                21,
                13,
                11,
                12,
                13,
                11,
                21,
                22,
                11,
                12,
                17,
                11,
                21,
                17,
                11,
                12,
                12,
                21,
                22,
                22,
                13,
                11,
                0,
            ],
        ],
        "id": ["0", "1", "2", "3"],
        "ner_tags": [
            [3, 0, 7, 0, 0, 0, 7, 0, 0],
            [1, 2],
            [5, 0],
            [
                0,
                3,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                7,
                0,
                0,
                0,
                0,
                0,
                7,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ],
        "pos_tags": [
            [22, 42, 16, 21, 35, 37, 16, 21, 7],
            [22, 22],
            [22, 11],
            [
                12,
                22,
                22,
                38,
                15,
                22,
                28,
                38,
                15,
                16,
                21,
                35,
                24,
                35,
                37,
                16,
                21,
                15,
                24,
                41,
                15,
                16,
                21,
                21,
                20,
                37,
                40,
                35,
                21,
                7,
            ],
        ],
        "tokens": [
            [
                "EU",
                "rejects",
                "German",
                "call",
                "to",
                "boycott",
                "British",
                "lamb",
                ".",
            ],
            ["Peter", "Blackburn"],
            ["BRUSSELS", "1996-08-22"],
            [
                "The",
                "European",
                "Commission",
                "said",
                "on",
                "Thursday",
                "it",
                "disagreed",
                "with",
                "German",
                "advice",
                "to",
                "consumers",
                "to",
                "shun",
                "British",
                "lamb",
                "until",
                "scientists",
                "determine",
                "whether",
                "mad",
                "cow",
                "disease",
                "can",
                "be",
                "transmitted",
                "to",
                "sheep",
                ".",
            ],
        ],
    }

    dev_data = {
        "chunk_tags": [
            [
                11,
                11,
                12,
                13,
                11,
                12,
                12,
                11,
                12,
                12,
                12,
                12,
                21,
                13,
                11,
                12,
                21,
                22,
                11,
                13,
                11,
                1,
                13,
                11,
                17,
                11,
                12,
                12,
                21,
                1,
                0,
            ],
            [
                0,
                11,
                21,
                22,
                22,
                11,
                12,
                12,
                17,
                11,
                21,
                22,
                22,
                11,
                12,
                13,
                11,
                0,
                0,
                11,
                12,
                11,
                12,
                12,
                12,
                12,
                12,
                12,
                21,
                11,
                12,
                12,
                0,
            ],
            [
                11,
                21,
                11,
                12,
                12,
                21,
                22,
                0,
                17,
                11,
                21,
                22,
                17,
                11,
                21,
                22,
                11,
                21,
                22,
                22,
                13,
                11,
                12,
                12,
                0,
            ],
            [
                11,
                21,
                11,
                12,
                11,
                12,
                13,
                11,
                12,
                12,
                12,
                12,
                21,
                22,
                11,
                12,
                0,
                11,
                0,
                11,
                12,
                13,
                11,
                12,
                12,
                12,
                12,
                12,
                21,
                11,
                12,
                1,
                2,
                2,
                11,
                21,
                22,
                11,
                12,
                0,
            ],
        ],
        "id": ["4", "5", "6", "7"],
        "ner_tags": [
            [
                5,
                0,
                0,
                0,
                0,
                3,
                4,
                0,
                0,
                0,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                3,
                0,
                0,
                0,
                1,
                2,
                2,
                2,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                3,
                0,
                0,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ],
        "pos_tags": [
            [
                22,
                27,
                21,
                35,
                12,
                22,
                22,
                27,
                16,
                21,
                22,
                22,
                38,
                15,
                22,
                24,
                20,
                37,
                21,
                15,
                24,
                16,
                15,
                22,
                15,
                12,
                16,
                21,
                38,
                17,
                7,
            ],
            [
                0,
                28,
                41,
                30,
                37,
                12,
                16,
                21,
                15,
                28,
                41,
                30,
                37,
                12,
                24,
                15,
                28,
                6,
                0,
                12,
                22,
                27,
                16,
                21,
                22,
                22,
                14,
                22,
                38,
                12,
                21,
                21,
                7,
            ],
            [
                28,
                38,
                16,
                16,
                21,
                38,
                40,
                10,
                15,
                28,
                38,
                40,
                15,
                21,
                38,
                40,
                28,
                20,
                37,
                40,
                15,
                12,
                22,
                22,
                7,
            ],
            [
                28,
                38,
                12,
                21,
                16,
                21,
                15,
                22,
                22,
                22,
                22,
                22,
                35,
                37,
                21,
                24,
                6,
                24,
                10,
                16,
                24,
                15,
                12,
                21,
                10,
                21,
                21,
                24,
                38,
                12,
                30,
                16,
                10,
                16,
                21,
                35,
                37,
                16,
                21,
                7,
            ],
        ],
        "tokens": [
            [
                "Germany",
                "'s",
                "representative",
                "to",
                "the",
                "European",
                "Union",
                "'s",
                "veterinary",
                "committee",
                "Werner",
                "Zwingmann",
                "said",
                "on",
                "Wednesday",
                "consumers",
                "should",
                "buy",
                "sheepmeat",
                "from",
                "countries",
                "other",
                "than",
                "Britain",
                "until",
                "the",
                "scientific",
                "advice",
                "was",
                "clearer",
                ".",
            ],
            [
                '"',
                "We",
                "do",
                "n't",
                "support",
                "any",
                "such",
                "recommendation",
                "because",
                "we",
                "do",
                "n't",
                "see",
                "any",
                "grounds",
                "for",
                "it",
                ",",
                '"',
                "the",
                "Commission",
                "'s",
                "chief",
                "spokesman",
                "Nikolaus",
                "van",
                "der",
                "Pas",
                "told",
                "a",
                "news",
                "briefing",
                ".",
            ],
            [
                "He",
                "said",
                "further",
                "scientific",
                "study",
                "was",
                "required",
                "and",
                "if",
                "it",
                "was",
                "found",
                "that",
                "action",
                "was",
                "needed",
                "it",
                "should",
                "be",
                "taken",
                "by",
                "the",
                "European",
                "Union",
                ".",
            ],
            [
                "He",
                "said",
                "a",
                "proposal",
                "last",
                "month",
                "by",
                "EU",
                "Farm",
                "Commissioner",
                "Franz",
                "Fischler",
                "to",
                "ban",
                "sheep",
                "brains",
                ",",
                "spleens",
                "and",
                "spinal",
                "cords",
                "from",
                "the",
                "human",
                "and",
                "animal",
                "food",
                "chains",
                "was",
                "a",
                "highly",
                "specific",
                "and",
                "precautionary",
                "move",
                "to",
                "protect",
                "human",
                "health",
                ".",
            ],
        ],
    }

    train_dataset = pd.DataFrame(train_data)
    dev_dataset = pd.DataFrame(dev_data)

    custom_sent_keys = ["tokens"]
    label_key = "ner_tags"

    X_train = train_dataset[custom_sent_keys]
    y_train = train_dataset[label_key]

    X_val = dev_dataset[custom_sent_keys]
    y_val = dev_dataset[label_key]

    automl = AutoML()

    automl_settings = {
        "gpu_per_trial": 0,
        "max_iter": 2,
        "time_budget": 5,
        "task": "token-classification",
        "metric": "seqeval",
    }

    automl_settings["custom_hpo_args"] = {
        "model_path": "bert-base-uncased",
        "output_dir": "test/data/output/",
        "ckpt_per_epoch": 5,
        "fp16": False,
    }

    try:
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **automl_settings
        )
    except requests.exceptions.HTTPError:
        return


if __name__ == "__main__":
    test_tokenclassification()

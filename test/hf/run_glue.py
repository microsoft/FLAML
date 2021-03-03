from flaml.nlp.finetune import AutoHuggingFace

def _test_electra(method='BlendSearch'):
    autohf = AutoHuggingFace()
    autohf.prepare_glue_data("resplit",
                             split_portion=
                             {"train": (0.0, 0.8),
                              "dev": (0.8, 0.9),
                              "test": (0.9, 1.0)})
    autohf.fit()

if __name__ == "__main__":
    _test_electra()
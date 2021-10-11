sentence_keys_glue = {
    "cola": ["sentence"],
    "mnli": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    "qnli": ["sentence", "question"],
    "qqp": ["question1", "question2"],
    "rte": ["sentence1", "sentence2"],
    "sst2": ["sentence"],
    "stsb": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}

sentence_keys_anli = ["hypothesis", "premise"]

sentence_keys_super_glue = {
    "rte": ["hypothesis", "premise"],
    "wic": ["sentence1", "sentence2"],
    "wsc": ["text"],
}

sentence_keys_hate_speech18 = ["text"]

sentence_keys_dbpedia_14 = ["content"]

sentence_keys_imdb = ["text"]

sentence_keys_yelp_review_full = ["text"]

sentence_keys_yelp_polarity = ["text"]

sentence_keys_amazon_polarity = ["content"]

sentence_keys_amazon_reviews_multi = ["review_body"]

sentence_keys_sentiment140 = ["review_body"]

sentence_keys_hyperpartisan_news_detection = ["title", "text"]


def get_sentence_keys(dataset_name, custom_sentence_keys=None):
    if custom_sentence_keys is not None:
        return custom_sentence_keys
    eval_name_mapping = globals()["sentence_keys_" + dataset_name]
    return eval_name_mapping

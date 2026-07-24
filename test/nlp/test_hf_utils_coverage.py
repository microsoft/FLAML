"""Tests to improve coverage for flaml/automl/nlp/huggingface/utils.py and flaml/automl/nlp/utils.py."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

transformers = pytest.importorskip("transformers", reason="transformers not installed")

from flaml.automl.task.task import (  # noqa: E402
    MULTICHOICECLASSIFICATION,
    SEQCLASSIFICATION,
    SEQREGRESSION,
    SUMMARIZATION,
    TOKENCLASSIFICATION,
)

# ---- Helpers ----


def _make_hf_args(**kwargs):
    defaults = {
        "max_seq_length": 32,
        "pad_to_max_length": False,
        "label_all_tokens": False,
        "label_list": ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def tokenizer():
    return _get_tokenizer()


# ==============================================================================
# Tests for flaml/automl/nlp/huggingface/utils.py
# ==============================================================================


class TestTodf:
    def test_todf_with_none_y(self):
        from flaml.automl.nlp.huggingface.utils import todf

        X = pd.DataFrame({"a": [1, 2]})
        result = todf(X, None, ["label"])
        assert result is None

    def test_todf_with_series(self):
        from flaml.automl.nlp.huggingface.utils import todf

        X = pd.DataFrame({"a": [1, 2]})
        Y = pd.Series([0, 1])
        result = todf(X, Y, ["label"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["label"]


class TestTokenizeText:
    """Cover lines 42-43, 45."""

    def test_token_classification_branch(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_text

        hf_args = _make_hf_args()
        X = pd.DataFrame({"tokens": [["John", "lives", "in", "Paris"], ["Hello", "world", "foo", "bar"]]})
        Y = pd.Series([[1, 0, 0, 3], [0, 0, 0, 0]])
        X_tok, Y_tok = tokenize_text(X, Y, task=TOKENCLASSIFICATION, hf_args=hf_args, tokenizer=tokenizer)
        assert X_tok is not None
        assert Y_tok is not None

    def test_nlg_branch(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_text

        hf_args = _make_hf_args()
        X = pd.DataFrame({"text": ["Hello world.", "Another sentence."]})
        Y = pd.Series(["Summary one.", "Summary two."], name="summary")
        X_tok, Y_tok = tokenize_text(X, Y, task=SUMMARIZATION, hf_args=hf_args, tokenizer=tokenizer)
        assert X_tok is not None

    def test_multichoice_branch(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_text

        hf_args = _make_hf_args()
        X = pd.DataFrame(
            {
                "sent1": ["A man is eating food."] * 2,
                "sent2": ["A man is"] * 2,
                "ending0": ["eating."] * 2,
                "ending1": ["sleeping."] * 2,
                "ending2": ["running."] * 2,
                "ending3": ["swimming."] * 2,
            }
        )
        Y = pd.Series([0, 1])
        X_tok, Y_tok = tokenize_text(X, Y, task=MULTICHOICECLASSIFICATION, hf_args=hf_args, tokenizer=tokenizer)
        assert X_tok is not None
        assert Y_tok is not None


class TestTokenizeSeq2Seq:
    """Cover lines 55, 62-64, 71, 75-76."""

    def test_with_y(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_seq2seq

        hf_args = _make_hf_args()
        X = pd.DataFrame({"text": ["Hello world.", "Test sentence."]})
        Y = pd.Series(["Hi.", "Test."], name="summary")
        inputs, outputs = tokenize_seq2seq(X, Y, tokenizer=tokenizer, task=SUMMARIZATION, hf_args=hf_args)
        assert inputs is not None
        assert outputs is not None
        assert "labels" in outputs.columns
        assert "input_ids" not in outputs.columns

    def test_without_y(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_seq2seq

        hf_args = _make_hf_args()
        X = pd.DataFrame({"text": ["Hello world.", "Test sentence."]})
        inputs, outputs = tokenize_seq2seq(X, None, tokenizer=tokenizer, task=SUMMARIZATION, hf_args=hf_args)
        assert inputs is not None
        assert outputs is None


class TestTokenizeAndAlignLabels:
    """Cover lines 90, 100-107, 112, 114, 116-125, 127."""

    def test_with_labels(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_and_align_labels

        hf_args = _make_hf_args()
        label_list = hf_args.label_list
        label_to_id = {i: i for i in range(len(label_list))}
        b_to_i_label = [0, 1, 2, 3, 4]

        examples = pd.Series({"tokens": ["John", "lives", "in", "Paris"], "tags": [1, 0, 0, 3]})
        result = tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            b_to_i_label=b_to_i_label,
            hf_args=hf_args,
            X_sent_key="tokens",
            Y_sent_key="tags",
            return_column_name=False,
        )
        assert isinstance(result, list)

    def test_with_labels_return_column_name(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_and_align_labels

        hf_args = _make_hf_args()
        label_to_id = {i: i for i in range(len(hf_args.label_list))}
        b_to_i_label = [0, 1, 2, 3, 4]

        examples = pd.Series({"tokens": ["John", "lives", "in", "Paris"], "tags": [1, 0, 0, 3]})
        result, col_names = tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            b_to_i_label=b_to_i_label,
            hf_args=hf_args,
            X_sent_key="tokens",
            Y_sent_key="tags",
            return_column_name=True,
        )
        assert "labels" in col_names

    def test_without_labels(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_and_align_labels

        hf_args = _make_hf_args()
        label_to_id = {i: i for i in range(len(hf_args.label_list))}
        b_to_i_label = [0, 1, 2, 3, 4]

        examples = pd.Series({"tokens": ["John", "lives"]})
        result = tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            b_to_i_label=b_to_i_label,
            hf_args=hf_args,
            X_sent_key="tokens",
            Y_sent_key=None,
            return_column_name=False,
        )
        assert isinstance(result, list)

    def test_label_all_tokens(self, tokenizer):
        """Cover line 112, 114 - label_all_tokens=True branch."""
        from flaml.automl.nlp.huggingface.utils import tokenize_and_align_labels

        hf_args = _make_hf_args(label_all_tokens=True)
        label_to_id = {i: i for i in range(len(hf_args.label_list))}
        # B-PER -> I-PER mapping
        b_to_i_label = [0, 2, 2, 4, 4]

        examples = pd.Series({"tokens": ["John", "lives", "in", "Paris"], "tags": [1, 0, 0, 3]})
        result, col_names = tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            b_to_i_label=b_to_i_label,
            hf_args=hf_args,
            X_sent_key="tokens",
            Y_sent_key="tags",
            return_column_name=True,
        )
        assert "labels" in col_names


class TestTokenizeTextTokClassification:
    """Cover lines 132-136, 138, 140-143, 145, 155, 168-172, 174, 176, 187, 200-204."""

    def test_with_y(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_text_tokclassification

        hf_args = _make_hf_args()
        X = pd.DataFrame({"tokens": [["John", "lives", "in", "Paris"], ["Hello", "world", "foo", "bar"]]})
        Y = pd.Series([[1, 0, 0, 3], [0, 0, 0, 0]])
        X_tok, Y_tok = tokenize_text_tokclassification(X, Y, tokenizer=tokenizer, hf_args=hf_args)
        assert X_tok is not None
        assert Y_tok is not None

    def test_without_y(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_text_tokclassification

        hf_args = _make_hf_args()
        X = pd.DataFrame({"tokens": [["John", "lives"], ["Hello", "world"]]})
        X_tok, Y_tok = tokenize_text_tokclassification(X, None, tokenizer=tokenizer, hf_args=hf_args)
        assert X_tok is not None
        assert Y_tok is None

    def test_b_to_i_label_mapping(self, tokenizer):
        """Cover lines 132-138: B-to-I label mapping with and without matching I- labels."""
        from flaml.automl.nlp.huggingface.utils import tokenize_text_tokclassification

        # Label list where B-PER has a matching I-PER but B-MISC does not have I-MISC
        hf_args = _make_hf_args(label_list=["O", "B-PER", "I-PER", "B-MISC"])
        X = pd.DataFrame({"tokens": [["John", "lives"], ["Hello", "world"]]})
        X_tok, Y_tok = tokenize_text_tokclassification(X, None, tokenizer=tokenizer, hf_args=hf_args)
        assert X_tok is not None


class TestTokenizeRow:
    """Cover lines 247, 257."""

    def test_with_prefix(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_row

        hf_args = _make_hf_args()
        row = ("Hello world.",)
        result = tokenize_row(
            row, tokenizer, prefix=("summarize: ",), task=SUMMARIZATION, hf_args=hf_args, return_column_name=False
        )
        assert isinstance(result, list)

    def test_nlg_decoder_input_ids(self, tokenizer):
        """Cover line 257: decoder_input_ids for NLG tasks."""
        from flaml.automl.nlp.huggingface.utils import tokenize_row

        hf_args = _make_hf_args()
        row = ("Hello world.",)
        result, cols = tokenize_row(
            row, tokenizer, prefix=None, task=SUMMARIZATION, hf_args=hf_args, return_column_name=True
        )
        assert "decoder_input_ids" in cols


class TestPostprocessPredictionAndTrue:
    """Cover lines 315, 321, 324-326, 333, 335-337, 341, 346-347, 349-350, 352-354, 356, 358-360, 362-366, 368, 370."""

    def test_y_pred_none(self, tokenizer):
        """Cover line 315."""
        from flaml.automl.nlp.huggingface.utils import postprocess_prediction_and_true

        hf_args = _make_hf_args()
        X = pd.DataFrame({"a": [1, 2, 3]})
        result, y_true = postprocess_prediction_and_true(SEQCLASSIFICATION, None, tokenizer, hf_args, X=X)
        assert len(result) == 3

    def test_seqclassification(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import postprocess_prediction_and_true

        hf_args = _make_hf_args()
        y_pred = np.array([[0.1, 0.9], [0.8, 0.2]])
        result, y_true = postprocess_prediction_and_true(SEQCLASSIFICATION, y_pred, tokenizer, hf_args)
        assert list(result) == [1, 0]

    def test_seqregression(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import postprocess_prediction_and_true

        hf_args = _make_hf_args()
        y_pred = np.array([[0.5], [0.7]])
        result, y_true = postprocess_prediction_and_true(SEQREGRESSION, y_pred, tokenizer, hf_args)
        assert result.shape == (2,)

    def test_tokenclassification_with_y_true(self, tokenizer):
        """Cover lines 321, 324, 335-337, 341, 346-347, 350."""
        from flaml.automl.nlp.huggingface.utils import postprocess_prediction_and_true

        hf_args = _make_hf_args()
        # 2 samples, 4 tokens, 5 labels
        y_pred = np.random.rand(2, 4, 5)
        y_true = pd.Series([[-100, 0, 1, -100], [-100, 0, 0, -100]])
        result, y_true_out = postprocess_prediction_and_true(
            TOKENCLASSIFICATION, y_pred, tokenizer, hf_args, y_true=y_true
        )
        assert isinstance(result, list)
        assert isinstance(y_true_out, list)
        # Only non -100 labels should remain
        assert len(result[0]) == 2
        assert len(y_true_out[0]) == 2

    def test_tokenclassification_without_y_true(self, tokenizer):
        """Cover lines 325-326, 333."""
        from flaml.automl.nlp.huggingface.utils import postprocess_prediction_and_true

        hf_args = _make_hf_args()
        y_pred = np.random.rand(2, 4, 5)
        X = pd.DataFrame({"tokens": [["John", "lives", "in", "Paris"], ["Hello", "world", "foo", "bar"]]})
        result, y_true_out = postprocess_prediction_and_true(
            TOKENCLASSIFICATION, y_pred, tokenizer, hf_args, y_true=None, X=X
        )
        assert isinstance(result, list)
        assert y_true_out is None

    def test_summarization_with_y_true(self, tokenizer):
        """Cover lines 352-354, 356, 358-360, 362-366."""
        from flaml.automl.nlp.huggingface.utils import postprocess_prediction_and_true

        hf_args = _make_hf_args()
        # Simulate logits: 2 samples, 5 tokens, vocab_size
        vocab_size = tokenizer.vocab_size
        y_pred_logits = np.random.rand(2, 5, vocab_size)
        y_true = np.array([[101, 2023, 2003, 102, -100], [101, 1037, 3231, 102, -100]])
        result, y_true_out = postprocess_prediction_and_true(
            SUMMARIZATION, (y_pred_logits,), tokenizer, hf_args, y_true=y_true
        )
        assert isinstance(result, list)
        assert isinstance(y_true_out, list)
        assert len(result) == 2

    def test_summarization_without_y_true(self, tokenizer):
        """Cover lines 367-368, 370."""
        from flaml.automl.nlp.huggingface.utils import postprocess_prediction_and_true

        hf_args = _make_hf_args()
        # y_pred as raw token ids (not tuple)
        y_pred = np.array([[101, 2023, 2003, 102, 0], [101, 1037, 3231, 102, 0]])
        result, y_true_out = postprocess_prediction_and_true(SUMMARIZATION, y_pred, tokenizer, hf_args, y_true=None)
        assert isinstance(result, list)
        assert y_true_out is None

    def test_multichoice(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import postprocess_prediction_and_true

        hf_args = _make_hf_args()
        y_pred = np.array([[0.1, 0.5, 0.3, 0.1], [0.4, 0.1, 0.1, 0.4]])
        y_true = np.array([1, 0])
        result, y_true_out = postprocess_prediction_and_true(
            MULTICHOICECLASSIFICATION, y_pred, tokenizer, hf_args, y_true=y_true
        )
        assert list(result) == [1, 0]


class TestLoadModel:
    """Cover lines 401, 403."""

    def test_load_model_token_classification(self):
        """Cover line 401 via mock."""
        from flaml.automl.nlp.huggingface.utils import load_model

        mock_config = MagicMock()
        mock_config.vocab_size = 30522

        mock_model = MagicMock()

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config), patch(
            "transformers.AutoModelForTokenClassification.from_pretrained", return_value=mock_model
        ):
            model = load_model("fake-path", task=TOKENCLASSIFICATION, num_labels=5)
            assert model is not None

    def test_load_model_nlg(self):
        """Cover line 403."""
        from flaml.automl.nlp.huggingface.utils import load_model

        mock_config = MagicMock()
        mock_config.vocab_size = 30522
        mock_model = MagicMock()

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config), patch(
            "transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=mock_model
        ):
            model = load_model("fake-path", task=SUMMARIZATION, num_labels=None)
            assert model is not None

    def test_load_model_multichoice(self):
        from flaml.automl.nlp.huggingface.utils import load_model

        mock_config = MagicMock()
        mock_config.vocab_size = 30522
        mock_model = MagicMock()

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config), patch(
            "transformers.AutoModelForMultipleChoice.from_pretrained", return_value=mock_model
        ):
            model = load_model("fake-path", task=MULTICHOICECLASSIFICATION, num_labels=None)
            assert model is not None


class TestTokenizeSwag:
    def test_tokenize_swag_return_column_name(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_swag

        hf_args = _make_hf_args()
        row = pd.Series(
            {
                "sent1": "A man is eating.",
                "sent2": "A man is",
                "ending0": "eating.",
                "ending1": "sleeping.",
                "ending2": "running.",
                "ending3": "swimming.",
            }
        )
        result, cols = tokenize_swag(row, tokenizer=tokenizer, hf_args=hf_args, return_column_name=True)
        assert isinstance(cols, list)

    def test_tokenize_swag_no_column_name(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_swag

        hf_args = _make_hf_args()
        row = pd.Series(
            {
                "sent1": "A man is eating.",
                "sent2": "A man is",
                "ending0": "eating.",
                "ending1": "sleeping.",
                "ending2": "running.",
                "ending3": "swimming.",
            }
        )
        result = tokenize_swag(row, tokenizer=tokenizer, hf_args=hf_args, return_column_name=False)
        assert isinstance(result, list)


class TestTokenizeOnedataframe:
    def test_with_pad_to_max_length(self, tokenizer):
        from flaml.automl.nlp.huggingface.utils import tokenize_onedataframe

        hf_args = _make_hf_args(pad_to_max_length=True)
        X = pd.DataFrame({"text": ["Hello world.", "Test."]})
        result = tokenize_onedataframe(X, tokenizer, task=SEQCLASSIFICATION, hf_args=hf_args, prefix_str="")
        assert result is not None


# ==============================================================================
# Tests for flaml/automl/nlp/utils.py
# ==============================================================================


class TestLoadDefaultMetric:
    """Cover lines 15-24."""

    def test_seqclassification(self):
        from flaml.automl.nlp.utils import load_default_huggingface_metric_for_task

        assert load_default_huggingface_metric_for_task(SEQCLASSIFICATION) == "accuracy"

    def test_seqregression(self):
        from flaml.automl.nlp.utils import load_default_huggingface_metric_for_task

        assert load_default_huggingface_metric_for_task(SEQREGRESSION) == "r2"

    def test_summarization(self):
        from flaml.automl.nlp.utils import load_default_huggingface_metric_for_task

        assert load_default_huggingface_metric_for_task(SUMMARIZATION) == "rouge1"

    def test_multichoice(self):
        from flaml.automl.nlp.utils import load_default_huggingface_metric_for_task

        assert load_default_huggingface_metric_for_task(MULTICHOICECLASSIFICATION) == "accuracy"

    def test_tokenclassification(self):
        from flaml.automl.nlp.utils import load_default_huggingface_metric_for_task

        assert load_default_huggingface_metric_for_task(TOKENCLASSIFICATION) == "seqeval"


class TestFormatVars:
    """Cover lines 43, 48."""

    def test_format_vars_basic(self):
        from flaml.automl.nlp.utils import format_vars

        resolved = {("learning_rate",): 0.001, ("batch_size",): 16}
        result = format_vars(resolved)
        assert "learning_rate" in result
        assert "batch_size" in result

    def test_format_vars_skip_run(self):
        """Cover line 43: skip 'run', 'env', 'resources_per_trial'."""
        from flaml.automl.nlp.utils import format_vars

        resolved = {("run", "x"): 1, ("env", "y"): 2, ("resources_per_trial", "z"): 3, ("lr",): 0.01}
        result = format_vars(resolved)
        assert "run" not in result
        assert "lr" in result

    def test_format_vars_with_int_keys(self):
        """Cover line 48: integer key in path."""
        from flaml.automl.nlp.utils import format_vars

        resolved = {("layers", 0, "size"): 128}
        result = format_vars(resolved)
        assert "0" in result


class TestLabelEncoderForTokenClassification:
    """Cover lines 95-98, 101-102, 105-107."""

    def test_fit_transform_with_string_labels(self):
        from flaml.automl.nlp.utils import LabelEncoderforTokenClassification

        encoder = LabelEncoderforTokenClassification()
        y = pd.Series([["O", "B-PER", "I-PER"], ["O", "O", "B-LOC"]])
        result = encoder.fit_transform(y)
        assert all(isinstance(v, int) for v in result.iloc[0])

    def test_fit_transform_with_int_labels(self):
        from flaml.automl.nlp.utils import LabelEncoderforTokenClassification

        encoder = LabelEncoderforTokenClassification()
        y = pd.Series([[0, 1, 2], [0, 0, 3]])
        result = encoder.fit_transform(y)
        assert list(result.iloc[0]) == [0, 1, 2]

    def test_transform_with_fitted_encoder(self):
        """Cover lines 105-107."""
        from flaml.automl.nlp.utils import LabelEncoderforTokenClassification

        encoder = LabelEncoderforTokenClassification()
        y_train = pd.Series([["O", "B-PER", "I-PER"], ["O", "O", "B-PER"]])
        encoder.fit_transform(y_train)
        y_test = pd.Series([["O", "B-PER"], ["I-PER", "O"]])
        result = encoder.transform(y_test)
        assert all(isinstance(v, int) for v in result.iloc[0])

    def test_transform_without_string_labels(self):
        """Cover line 105-107 else path: no _tokenlabel_to_id."""
        from flaml.automl.nlp.utils import LabelEncoderforTokenClassification

        encoder = LabelEncoderforTokenClassification()
        y = pd.Series([[0, 1, 2], [0, 0, 3]])
        encoder.fit_transform(y)
        result = encoder.transform(y)
        assert list(result.iloc[0]) == [0, 1, 2]

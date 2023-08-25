from flaml.autogen.token_count_utils import get_max_token_limit, count_token, percentile_used, token_left


def test_token_count():
    assert get_max_token_limit("gpt-3.5-turbo-0613") == 4096
    assert count_token("\n" + "Error: The output exceeds the length limit and is truncated.", model="gpt-4") == 13
    assert (
        token_left("\n" + "Error: The output exceeds the length limit and is truncated.", model="gpt-3.5-turbo-0613")
        == 4083
    )
    assert (
        percentile_used(
            "\n" + "Error: The output exceeds the length limit and is truncated.", model="gpt-3.5-turbo-0613"
        )
        == 0.003173828125
    )


if __name__ == "__main__":
    test_token_count()

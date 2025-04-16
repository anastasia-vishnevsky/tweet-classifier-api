import pytest
from src.utils.text_cleaner import TextCleaner

@pytest.fixture
def text_cleaner():
    return TextCleaner()

@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Bought 5 apples for $10", "buy apple for"),
        ("This is so COOL!!! 😎🔥", "this be so cool"),
        ("LoVe ThIs SONG", "love this song"),
        ("Had pizza 🍕 and slept 😴", "have pizza and sleep"),
        ("LOL, IDK what's going on tbh", "lol idk what go on tbh"),
        ("     \n\t", ""),
        ("WOW WOW WOW", "wow wow wow"),
    ]
)
def test_clean_text(text_cleaner, input_text, expected_output):
    assert text_cleaner.clean_text(input_text) == expected_output
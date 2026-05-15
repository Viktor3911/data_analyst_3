from analytics_agent.config import parse_model_list, parse_positive_int


def test_parse_model_list_deduplicates_and_strips_values() -> None:
    assert parse_model_list(" a, b, a ,, c ") == ["a", "b", "c"]


def test_parse_positive_int_returns_default_for_invalid_values() -> None:
    assert parse_positive_int("15", default=30) == 15
    assert parse_positive_int("0", default=30) == 30
    assert parse_positive_int("bad", default=30) == 30

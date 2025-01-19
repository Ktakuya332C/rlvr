from rlvr.loaders import get_gsm8k


def test_get_gsm8k():
    dataset = get_gsm8k()
    item = next(iter(dataset.iter_rows()))
    assert set(item.keys()) == {"question", "answer"}

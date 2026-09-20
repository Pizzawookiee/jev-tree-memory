from src.split import split_sentences


def test_offsets_and_protected_periods():
    text = "Dr. Lee paid 3.14 at https://example.com. Email a.b@example.com! Next item."
    spans = split_sentences(text)
    assert [text[start:end] for _, start, end in spans] == [item for item, _, _ in spans]
    assert len(spans) == 3
    assert spans[0][0].startswith("Dr. Lee paid 3.14")


def test_long_sentence_is_losslessly_chunked():
    text = "word " * 200
    spans = split_sentences(text, maximum=80)
    assert all(len(value) <= 80 for value, _, _ in spans)
    assert all(text[start:end] == value for value, start, end in spans)


from brain.speech_text import SentenceChunker, clean_for_speech


def stream(text, chunker=None, step=3):
    c = chunker or SentenceChunker()
    out = []
    for i in range(0, len(text), step):
        out += list(c.feed(text[i:i + step]))
    return out + list(c.flush())


def test_markdown_is_stripped():
    md = "## Plan\n- **Bold** item with `code`\n1. See [the docs](https://x.y) or https://a.b/c 🚀"
    cleaned = clean_for_speech(md)
    for junk in ("#", "*", "`", "](", "https", "🚀", "- "):
        assert junk not in cleaned
    assert "Bold item with code" in cleaned
    assert "the docs" in cleaned


def test_sentences_emitted_as_they_complete():
    c = SentenceChunker()
    assert list(c.feed("Hello Reza. How")) == ["Hello Reza."]
    assert list(c.feed(" are you?")) == []          # no trailing space yet
    assert list(c.flush()) == ["How are you?"]


def test_abbreviations_and_decimals_do_not_split():
    out = stream("Dr. Smith measured 3.5 metres. Done.")
    assert out == ["Dr. Smith measured 3.5 metres.", "Done."]


def test_long_first_sentence_splits_at_first_clause():
    text = ("Reza, octopuses are highly intelligent creatures, and they have been observed "
            "opening jars to eat what is inside.")
    out = stream(text)
    assert out[0] == "Reza, octopuses are highly intelligent creatures,"
    assert " ".join(out) == text


def test_later_text_only_splits_when_long():
    c = SentenceChunker()
    out = stream("Fine. Short, and sweet.", c)
    assert out == ["Fine.", "Short, and sweet."]


def test_bullets_become_separate_clean_chunks():
    out = stream("Two things:\n- First one\n- Second one")
    assert out == ["Two things:", "First one", "Second one"]

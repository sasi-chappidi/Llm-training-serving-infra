from src.tokenizer import CharTokenizer


def test_encode_decode_roundtrip():
    text = "hello world"
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode("hello")
    decoded = tokenizer.decode(ids)
    assert decoded == "hello"
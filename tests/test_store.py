"""store.py：Chroma 写入与检索的往返行为。"""

from app.rag.models import TextChunk


def test_add_and_search_roundtrip_with_metadata(store, collection):
    chunks = [TextChunk(id="a:0", text="hello world", source_name="a.pdf", page=2, heading="Intro")]
    store.add_chunks(collection, chunks, [[0.5, 0.5]])
    hits = store.search(collection, [0.5, 0.5], top_k=3)
    assert len(hits) == 1
    text, meta, sim = hits[0]
    assert text == "hello world"
    assert meta["source_name"] == "a.pdf"
    assert meta["page"] == 2
    assert meta["heading"] == "Intro"
    assert sim > 0.99


def test_search_empty_collection_returns_empty(store, collection):
    assert store.search(collection, [1.0, 0.0]) == []


def test_search_returns_most_similar_first(store, collection):
    dim = 5
    chunks, embeddings = [], []
    for i in range(dim):
        emb = [0.0] * dim
        emb[i] = 1.0
        chunks.append(TextChunk(id=f"b:{i}", text=f"chunk {i}", source_name="b.txt"))
        embeddings.append(emb)
    store.add_chunks(collection, chunks, embeddings)
    hits = store.search(collection, embeddings[2], top_k=5)
    assert hits[0][0] == "chunk 2"

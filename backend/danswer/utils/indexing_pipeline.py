from collections.abc import Callable
from functools import partial
from itertools import chain

from danswer.chunking.chunk import Chunker
from danswer.chunking.chunk import DefaultChunker
from danswer.chunking.models import EmbeddedIndexChunk
from danswer.connectors.models import Document
from danswer.datastores.interfaces import Datastore
from danswer.datastores.qdrant.store import QdrantDatastore
from danswer.embedding.biencoder import DefaultEmbedder
from danswer.embedding.type_aliases import Embedder


def _indexing_pipeline(
    chunker: Chunker,
    embedder: Embedder,
    datastore: Datastore,
    documents: list[Document],
) -> list[EmbeddedIndexChunk]:
    # TODO: make entire indexing pipeline async to not block the entire process
    # when running on async endpoints
    chunks = list(chain(*[chunker.chunk(document) for document in documents]))
    chunks_with_embeddings = embedder.embed(chunks)
    datastore.index(chunks_with_embeddings)
    return chunks_with_embeddings


def build_indexing_pipeline(
    *,
    chunker: Chunker | None = None,
    embedder: Embedder | None = None,
    datastore: Datastore | None = None,
) -> Callable[[list[Document]], list[EmbeddedIndexChunk]]:
    """Builds a pipline which takes in a list of docs and indexes them.

    Default uses _ chunker, _ embedder, and qdrant for the datastore"""
    if chunker is None:
        chunker = DefaultChunker()

    if embedder is None:
        embedder = DefaultEmbedder()

    if datastore is None:
        datastore = QdrantDatastore()

    return partial(_indexing_pipeline, chunker, embedder, datastore)

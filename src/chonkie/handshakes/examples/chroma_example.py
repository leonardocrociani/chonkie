"""ChromaDB Handshake Example."""

import os

import chromadb

from chonkie.chunker import TokenChunker
from chonkie.handshakes.vector_db_writers import ChromaHandshake
from chonkie.refinery import EmbeddingsRefinery
from chonkie.types import Chunk

# 1. Define sample text
sample_text = "This is a sample text to test the ChromaDB handshake. It contains multiple sentences and should be chunked and embedded."


# 2. Instantiate and use TokenChunker
print("Chunking text...")
token_chunker = TokenChunker(
    chunk_size=20, chunk_overlap=5, return_type="chunks"
)
chunks: list[Chunk] = token_chunker(sample_text)  # type: ignore
print(f"Created {len(chunks)} chunks.")


# 3. Instantiate and use EmbeddingRefinery
print("Embedding chunks...")
# Use a simple, locally available model
embedding_refinery = EmbeddingsRefinery(embedding_model="minishlab/potion-base-8M")
refined_chunks = embedding_refinery(chunks)
print("Embeddings added to chunks.")


# 4. Instantiate ChromaDB client
print("Connecting to ChromaDB...")
# Ensure you have the Chroma API key securely managed, e.g., via environment variables
# For this example, using the provided key directly, but replace with secure handling in production
chroma_api_key = os.getenv(
    "CHROMA_API_KEY", None
)  # Replace with your actual key or env var
if not chroma_api_key:
    raise ValueError(
        "Chroma API key not found. Set the CHROMA_API_KEY environment variable."
    )

client = chromadb.HttpClient(
    ssl=True,
    host="api.trychroma.com",
    tenant="f5e9a8f6-f386-4b8b-8f7a-5f7ea03bdad9",
    database="chonkie-writer-test",
    headers={"x-chroma-token": chroma_api_key},
)
print("Connected to ChromaDB.")

# 5. Instantiate and use ChromaHandshake
collection_name = "chonkie_test_collection"
print(f"Using collection: {collection_name}")
chroma_handshake = ChromaHandshake(
    client=client, collection_name=collection_name
)

# Ensure the collection exists (or create it)
# This might be handled within the handshake or might need explicit handling
# depending on ChromaHandshake's implementation. Assuming it handles creation/getting.
# Let's explicitly get or create for clarity:
print(f"Getting or creating collection '{collection_name}'...")
collection = client.get_or_create_collection(collection_name)
print("Collection ready.")


# 6. Write chunks to ChromaDB
print("Writing chunks to ChromaDB...")
try:
    chroma_handshake.write(refined_chunks)
    print(
        f"Successfully wrote {len(refined_chunks)} chunks to ChromaDB collection '{collection_name}'."
    )

    # Optional: Verify by querying
    print("Verifying write by querying...")
    results = collection.get(
        ids=[c.id for c in refined_chunks], include=["metadatas", "documents"]
    )
    print(f"Retrieved {len(results.get('ids', []))} items from Chroma.")
    # print("Retrieved data:", results)

except Exception as e:
    print(f"An error occurred during write: {e}")

print("Script finished.")

#!/usr/bin/env python3
"""PostgreSQL + pgvector with PsycopgHandshake Example.

===================================================

This example demonstrates how to use Chonkie's PsycopgHandshake to store and 
search text chunks in PostgreSQL with pgvector for semantic similarity search.
We'll chunk a sample document, store it in PostgreSQL, and perform vector 
similarity searches.

Prerequisites:
- PostgreSQL running with pgvector extension
- Install dependencies: pip install "chonkie[psycopg]"
- Set up your PostgreSQL connection details

What you'll learn:
1. How to connect to PostgreSQL with psycopg3
2. Store chunked text with embeddings using PsycopgHandshake
3. Perform semantic similarity searches
4. Create vector indexes for better performance
5. Handle different distance metrics (L2, cosine, inner product)

Author: Chonkie Team
Date: 2025

"""

import os
import sys
import time
from typing import Dict, List

try:
    import psycopg
except ImportError:
    print("❌ Required dependencies not found!")
    print("Install with: pip install 'chonkie[psycopg]'")
    sys.exit(1)

from chonkie import AutoEmbeddings, PsycopgHandshake, RecursiveChunker
from chonkie.types import Chunk


def setup_database_connection() -> psycopg.Connection:
    """Set up PostgreSQL connection with pgvector support."""
    # Database connection parameters
    # You can modify these or use environment variables
    db_config: Dict[str, str] = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "chonkie_demo"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    }
    
    print("🔌 Connecting to PostgreSQL...")
    print(f"   Host: {db_config['host']}:{db_config['port']}")
    print(f"   Database: {db_config['dbname']}")
    print(f"   User: {db_config['user']}")
    
    try:
        # Create connection
        connection = psycopg.connect(**db_config)  # type: ignore
        
        # Test connection
        with connection.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]  # type: ignore
            print(f"✅ Connected to PostgreSQL: {version[:50]}...")
            
        return connection
        
    except psycopg.Error as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure PostgreSQL is running")
        print("   2. Check your connection parameters")
        print("   3. Ensure the database exists")
        print("   4. Verify user permissions")
        print("\n📚 Quick setup commands:")
        print("   createdb chonkie_demo")
        print("   psql -d chonkie_demo -c 'CREATE EXTENSION vector;'")
        sys.exit(1)


if __name__ == "__main__":
    """
    Example demonstrating PsycopgHandshake with PostgreSQL and pgvector.
    
    This example shows how to:
    1. Connect to PostgreSQL with pgvector
    2. Initialize PsycopgHandshake with embeddings
    3. Chunk and store documents with vector embeddings
    4. Perform semantic similarity searches
    5. Create indexes for optimal performance
    """
    
    print("🦛 Chonkie + PostgreSQL + pgvector Example")
    print("=" * 50)
    
    # Sample text - research paper abstract about vector databases
    sample_text = """
    Vector databases have emerged as a critical infrastructure component for modern 
    AI applications, particularly those involving large language models and semantic 
    search capabilities. These specialized databases are designed to efficiently 
    store, index, and query high-dimensional vector embeddings that represent 
    semantic meaning of text, images, or other data types.
    
    The core challenge in vector databases lies in the curse of dimensionality, 
    where traditional indexing methods become inefficient as the number of dimensions 
    increases. To address this, vector databases employ sophisticated indexing 
    algorithms such as Hierarchical Navigable Small World (HNSW) graphs, Inverted 
    File (IVF) indexes, and Product Quantization (PQ) techniques.
    
    HNSW is particularly effective for approximate nearest neighbor search, 
    constructing a multi-layer graph structure that enables logarithmic search 
    complexity. IVF indexes partition the vector space into clusters, allowing 
    for faster search by limiting the search scope to relevant clusters. Product 
    Quantization reduces memory usage by compressing high-dimensional vectors 
    into compact representations.
    
    Modern vector databases like Pinecone, Weaviate, and Qdrant provide cloud-native 
    solutions with features such as real-time updates, horizontal scaling, and 
    integration with machine learning pipelines. Open-source alternatives like 
    Chroma and FAISS offer flexibility for on-premises deployments and research 
    applications.
    
    The integration of vector databases with traditional relational databases, 
    such as PostgreSQL with the pgvector extension, provides a hybrid approach 
    that combines the benefits of structured data management with vector similarity 
    search capabilities. This enables developers to build applications that can 
    handle both traditional queries and semantic search within a single system.
    
    As AI applications continue to evolve, vector databases will play an increasingly 
    important role in enabling semantic search, recommendation systems, retrieval-
    augmented generation (RAG), and other AI-powered features that require efficient 
    similarity search over high-dimensional data.
    """
    
    # Step 1: Set up database connection
    print("\n🔧 Step 1: Database Connection")
    print("-" * 30)
    connection = setup_database_connection()
    
    # Step 2: Initialize embeddings model
    print("\n🔧 Step 2: Loading Embedding Model")
    print("-" * 30)
    
    # Use a fast, lightweight model for this example
    print("🔄 Loading Model2Vec embeddings (fast and lightweight)...")
    embeddings = AutoEmbeddings.get_embeddings("minishlab/potion-base-8M")
    print(f"✅ Loaded embeddings model: {embeddings}")
    print(f"   Dimensions: {embeddings.dimension}")
    
    # Step 3: Initialize PsycopgHandshake
    print("\n🔧 Step 3: Setting up PsycopgHandshake")
    print("-" * 30)
    
    handshake = PsycopgHandshake(
        connection=connection,
        table_name="vector_chunks",
        embedding_model=embeddings,
        create_table=True  # This will create the table if it doesn't exist
    )
    print("✅ PsycopgHandshake initialized:")
    print(f"   Table: {handshake.table_name}")
    print(f"   Vector dimensions: {handshake.vector_dimensions}")
    
    # Step 4: Set up chunker and chunk the text
    print("\n🔧 Step 4: Chunking the Document")
    print("-" * 30)
    
    chunker = RecursiveChunker(
        chunk_size=200,  # Smaller chunks for better granularity
        min_characters_per_chunk=50  # Minimum characters per chunk
    )
    
    chunks: List[Chunk] = chunker.chunk(sample_text)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Display chunk information
    for i, chunk in enumerate(chunks):
        print(f"\n📄 Chunk {i+1}:")
        print(f"   Length: {len(chunk.text)} characters")
        print(f"   Token count: {chunk.token_count}")
        print(f"   Preview: {chunk.text[:80]}...")
    
    # Step 5: Store chunks in PostgreSQL
    print("\n🔧 Step 5: Storing Chunks in PostgreSQL")
    print("-" * 30)
    
    print("🔄 Generating embeddings and storing chunks...")
    chunk_ids = handshake.write(chunks)
    print(f"✅ Stored {len(chunk_ids)} chunks in PostgreSQL")
    print(f"   Sample IDs: {chunk_ids[:3]}...")
    
    # Step 6: Perform similarity searches
    print("\n🔧 Step 6: Semantic Similarity Search")
    print("-" * 30)
    
    # Test different search queries
    search_queries = [
        "What are HNSW graphs and how do they work?",
        "Tell me about vector indexing algorithms",
        "How does PostgreSQL integrate with vector search?",
        "What are the benefits of cloud-native vector databases?"
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n🔍 Search Query {i}: '{query}'")
        print("-" * 40)
        
        # Search with different distance metrics
        for metric in ["l2", "cosine"]:
            print(f"\n📊 Results using {metric.upper()} distance:")
            results = handshake.search(
                query=query, 
                limit=3, 
                distance_metric=metric
            )
            
            for j, result in enumerate(results, 1):
                print(f"   {j}. Distance: {result['distance']:.4f}")
                print(f"      Text: {result['text'][:100]}...")
                print(f"      Tokens: {result['token_count']}")
    
    # Step 7: Create indexes for better performance
    print("\n🔧 Step 7: Creating Vector Indexes")
    print("-" * 30)
    
    print("🔄 Creating HNSW index for L2 distance...")
    handshake.create_index(
        index_type="hnsw",
        distance_metric="l2",
        m=16,  # Number of connections in HNSW graph
        ef_construction=64  # Size of dynamic candidate list
    )
    
    print("🔄 Creating IVFFlat index for cosine similarity...")
    handshake.create_index(
        index_type="ivfflat",
        distance_metric="cosine",
        lists=100  # Number of clusters
    )
    
    # Step 8: Demonstrate advanced search features
    print("\n🔧 Step 8: Advanced Search Examples")
    print("-" * 30)
    
    # Search for technical concepts
    technical_query = "machine learning algorithms clustering"
    print(f"\n🎯 Technical search: '{technical_query}'")
    
    technical_results = handshake.search(
        query=technical_query,
        limit=2,
        distance_metric="cosine"
    )
    
    for i, result in enumerate(technical_results, 1):
        print(f"\n📋 Result {i}:")
        print(f"   ID: {result['id']}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Chunk Type: {result['metadata'].get('chunk_type', 'Unknown')}")
        print(f"   Position: {result['start_index']}-{result['end_index']}")
        print(f"   Text: {result['text'][:200]}...")
    
    # Step 9: Performance comparison
    print("\n🔧 Step 9: Performance Comparison")
    print("-" * 30)
    
    performance_query = "vector database indexing performance"
    
    # Time the search operation
    start_time = time.time()
    perf_results = handshake.search(
        query=performance_query,
        limit=5,
        distance_metric="l2"
    )
    search_time = time.time() - start_time
    
    print("⚡ Search performance:")
    print(f"   Query: '{performance_query}'")
    print(f"   Results: {len(perf_results)}")
    print(f"   Search time: {search_time:.4f} seconds")
    print(f"   Avg time per result: {(search_time/len(perf_results)*1000):.2f} ms")
    
    # Step 10: Clean up and summary
    print("\n🔧 Step 10: Summary and Cleanup")
    print("-" * 30)
    
    # Get table statistics
    with connection.cursor() as cur:
        cur.execute(f"""
            SELECT 
                COUNT(*) as total_chunks,
                AVG(LENGTH(text)) as avg_text_length,
                MIN(token_count) as min_tokens,
                MAX(token_count) as max_tokens,
                AVG(token_count) as avg_tokens
            FROM {handshake.table_name}
        """)
        stats = cur.fetchone()

    if stats:
        print("📊 Final Statistics:")
        print(f"   Total chunks stored: {stats[0]}")
        print(f"   Average text length: {stats[1]:.1f} characters")
        print(f"   Token range: {stats[2]} - {stats[3]}")
        print(f"   Average tokens per chunk: {stats[4]:.1f}")
    else:
        print("📊 No statistics available (no data found)")
    
    # Optional: Clean up the table (uncomment if desired)
    # print("\n🧹 Cleaning up...")
    # with connection.cursor() as cur:
    #     cur.execute(f"DROP TABLE IF EXISTS {handshake.table_name}")
    # connection.commit()
    # print("✅ Table dropped")
    
    # Close connection
    connection.close()
    print("✅ Database connection closed")
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Next steps you could try:")
    print("   - Experiment with different embedding models (OpenAI, Cohere, etc.)")
    print("   - Try different chunking strategies (SentenceChunker, SemanticChunker)")
    print("   - Implement a full RAG (Retrieval-Augmented Generation) pipeline")
    print("   - Integrate with your existing PostgreSQL schema")
    print("   - Scale up with larger documents and datasets")
    print("   - Compare performance with different index configurations")
    print(f"   - Query the stored data directly: SELECT * FROM {handshake.table_name} LIMIT 5;")
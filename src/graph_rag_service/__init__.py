"""
Graph-RAG Service

This service handles:
- Index documents into OpenSearch and Qdrant
- Serve hybrid BM25 + vector + graph retrieval
- Return answers with citations and graph context
"""
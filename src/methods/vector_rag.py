import chromadb
from chromadb.utils import embedding_functions
import uuid

class VectorRAG:
    name = "Vector RAG"
    def __init__(self, model, top_k: int = 3, chunk_size: int = 500, document: str = None):
        self.model = model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.client = chromadb.Client()
        # Use default embedding function (all-MiniLM-L6-v2)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        
        self.current_document = None
        self.current_collection_name = None
        
        if document:
            self._index_document(document)
        
    def _chunk_text(self, text):
        # Simple character-based chunking with overlap
        chunks = []
        metadatas = []
        overlap = 50
        step = max(1, self.chunk_size - overlap)
        for i in range(0, len(text), step):
            chunk_text = text[i:i+self.chunk_size]
            chunks.append(chunk_text)
            metadatas.append({"start": i, "end": i + len(chunk_text)})
        return chunks, metadatas

    def _index_document(self, document: str):
        # Delete previous collection if it exists
        if self.current_collection_name:
            try:
                self.client.delete_collection(self.current_collection_name)
            except ValueError:
                pass # Collection might not exist
        
        self.current_collection_name = f"doc_{uuid.uuid4().hex}"
        collection = self.client.create_collection(name=self.current_collection_name, embedding_function=self.ef)
        
        chunks, metadatas = self._chunk_text(document)
        if not chunks:
             # Handle empty document case
             self.current_document = document
             return

        ids = [str(i) for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        self.current_document = document

    def answer(self, question: str, document: str = None) -> dict:
        # If document is provided and different, re-index
        if document is not None and document != self.current_document:
            self._index_document(document)
        
        # If no document indexed yet (and none provided), return empty
        if not self.current_collection_name:
             return {"answer": "", "retrieved_chunks": []}

        try:
            collection = self.client.get_collection(name=self.current_collection_name, embedding_function=self.ef)
        except ValueError:
             # Should not happen unless manually deleted, but safe fallback
             if document:
                 self._index_document(document)
                 collection = self.client.get_collection(name=self.current_collection_name, embedding_function=self.ef)
             else:
                 return {"answer": "", "retrieved_chunks": []}
        
        # Check if collection is empty (e.g. empty doc)
        if collection.count() == 0:
             return {"answer": "", "retrieved_chunks": []}

        results = collection.query(query_texts=[question], n_results=min(collection.count(), self.top_k))
        
        retrieved_texts = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(retrieved_texts)
        
        # Combine text and metadata
        retrieved_chunks = []
        for text, meta in zip(retrieved_texts, retrieved_metadatas):
            retrieved_chunks.append({"text": text, "metadata": meta})
        
        context = "\n\n".join(retrieved_texts)
        ans = self.model.ask(question, context)
        
        # Do NOT delete collection here
        
        return {"answer": ans, "retrieved_chunks": retrieved_chunks}

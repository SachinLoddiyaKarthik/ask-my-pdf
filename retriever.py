import os
import json
import hashlib
import numpy as np
import logging
from typing import List, Dict, Optional, Set

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Configure logging
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, llm_handler):
        """
        Initializes the Retriever with an LLMHandler instance to get embedding models.
        """
        self.llm_handler = llm_handler

    def reciprocal_rank_fusion(self, results: List[List[Document]], k=60) -> List[Document]:
        """
        Combines multiple ranked lists of documents using Reciprocal Rank Fusion.
        RRF gives higher scores to documents that appear in the top ranks of multiple lists,
        effectively merging the strengths of different retrieval methods.
        """
        fused_scores = {}
        
        # Calculate RRF scores for each document
        for ranks in results:
            for rank, doc in enumerate(ranks):
                doc_content = doc.page_content
                if doc_content not in fused_scores:
                    fused_scores[doc_content] = 0
                fused_scores[doc_content] += 1 / (k + rank)
        
        # Sort documents by their fused scores in descending order
        unique_reranked_docs = []
        seen_content = set() # To ensure uniqueness of documents in the final list
        
        for doc_content, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
            doc_obj = None
            # Find the original Document object corresponding to the content
            for sublist in results:
                for doc in sublist:
                    if doc.page_content == doc_content:
                        doc_obj = doc
                        break
                if doc_obj:
                    break
            
            # Add to the final list if it's a unique document
            if doc_obj and doc_content not in seen_content:
                unique_reranked_docs.append(doc_obj)
                seen_content.add(doc_obj.page_content)
                
        return unique_reranked_docs

    # Removed embeddings_model from the signature as it's not directly used here
    def rag_fusion_retrieve(self, query: str, vectorstore, bm25_retriever, k_vector=5, k_bm25=5) -> List[Document]:
        """
        Performs RAG Fusion by combining vector search (semantic) and BM25 search (keyword) results.
        The results from both methods are then combined using Reciprocal Rank Fusion.
        """
        # Vector Search (semantic similarity)
        # vectorstore.similarity_search uses the embedding function it was initialized with
        vector_results = vectorstore.similarity_search(query, k=k_vector)

        # BM25 Search (keyword matching)
        tokenized_query = query.split()
        bm25_scores = bm25_retriever.get_scores(tokenized_query)
        
        # Get all documents from the vectorstore's docstore for BM25 ranking
        all_bm25_docs = [doc for doc in vectorstore.docstore._dict.values()]
        
        # Sort documents by BM25 scores in descending order
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
        
        bm25_results = []
        seen_content_bm25 = set() # To ensure uniqueness of documents in BM25 results
        for idx in bm25_ranked_indices:
            if idx < len(all_bm25_docs):
                doc = all_bm25_docs[idx]
                if doc.page_content not in seen_content_bm25:
                    bm25_results.append(doc)
                    seen_content_bm25.add(doc.page_content)
                if len(bm25_results) >= k_bm25: # Limit to k_bm25 results
                    break
        
        # Combine results from both search methods using RRF
        return self.reciprocal_rank_fusion([vector_results, bm25_results])

    class CustomRagFusionRetriever(BaseRetriever):
        """
        A custom LangChain retriever that integrates RAG Fusion logic.
        This class allows the RAG Fusion retrieval mechanism to be used directly
        within LangChain's conversational chains.
        """
        vectorstore: any
        bm25: any
        # Removed embeddings_model from attributes as it's not needed directly here
        parent_retriever: any # Reference to the main Retriever instance

        def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            """
            Method required by BaseRetriever to get relevant documents for a given query.
            Delegates to the rag_fusion_retrieve method of the parent Retriever instance.
            """
            return self.parent_retriever.rag_fusion_retrieve(
                query, self.vectorstore, self.bm25 # Removed embeddings_model argument
            )

    def create_vector_store(self, processed_data: Dict, chunk_size: int, chunk_overlap: int) -> str:
        """
        Creates a deduplicated FAISS vector store from text chunks and saves it locally.
        Generates a unique store ID based on the content hash and saves metadata.
        """
        text = processed_data['text']
        metadata = processed_data['metadata']

        # Moved chunking logic here, as it's part of vector store creation prep
        from pdf_utils import PDFUtils
        pdf_util_instance = PDFUtils()
        chunks = pdf_util_instance.create_text_chunks(text, chunk_size, chunk_overlap)
        
        if not chunks:
            raise ValueError("Cannot create vector store: no valid text chunks generated.")

        # Deduplicate chunks to avoid redundant embeddings and improve efficiency
        seen = set()
        unique_chunks = [chunk for chunk in chunks if not (chunk in seen or seen.add(chunk))]
        
        embeddings = self.llm_handler.get_embeddings_model()
        vectorstore = FAISS.from_texts(unique_chunks, embeddings)
        
        # Create a unique ID for the document set based on its content
        store_id = hashlib.sha256("".join(unique_chunks).encode()).hexdigest()
        store_name = f"docset_{store_id}"
        store_path = os.path.join("faiss_indexes", store_name)
        
        os.makedirs(store_path, exist_ok=True)
        vectorstore.save_local(store_path) # Save the FAISS index
        
        metadata['unique_chunks'] = len(unique_chunks)
        metadata['original_chunks'] = len(chunks)
        
        # Save metadata and an empty chat history file alongside the vector store
        with open(os.path.join(store_path, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        with open(os.path.join(store_path, "chat_history.json"), 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False) # Initialize empty chat history

        return store_name

    def load_vector_store(self, store_name: str):
        """
        Loads a FAISS vector store and its associated metadata and BM25 index.
        Reconstructs the BM25 index from the documents stored in the FAISS index.
        """
        store_path = os.path.join("faiss_indexes", store_name)
        if not os.path.exists(store_path):
            logger.error(f"Vector store directory '{store_path}' not found.")
            return None, None, None
            
        embeddings = self.llm_handler.get_embeddings_model()
        # allow_dangerous_deserialization=True is necessary for loading custom classes like Document
        vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Load metadata
        metadata = {}
        metadata_path = os.path.join(store_path, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata for {store_name}: {e}")

        # Re-create BM25 index from the loaded vectorstore's documents
        # The docstore._dict contains the Document objects indexed in FAISS
        chunks = [doc.page_content for doc in vectorstore.docstore._dict.values()]
        bm25 = BM25Okapi([doc.split() for doc in chunks]) # BM25 expects tokenized documents
        
        return vectorstore, bm25, metadata

    def merge_vector_stores(self, store_names: Set[str]):
        """
        Merges multiple FAISS vector stores and their BM25 indexes into a single usable unit.
        Handles duplicate chunks and warns about inconsistent indexing languages.
        This is crucial for multi-document querying.
        """
        if not store_names:
            return None, None

        main_store = None
        main_bm25_documents = [] # List to collect all unique tokenized documents for the merged BM25
        indexed_languages = set()
        seen_chunks_for_merge = set() # To prevent adding duplicate documents to main_store

        for name in store_names:
            current_store, current_bm25, metadata = self.load_vector_store(name)
            if not current_store:
                logger.warning(f"Skipping '{name}' as it could not be loaded.")
                continue
            
            # Track indexed languages to warn about inconsistencies
            indexed_languages.add(metadata.get('indexed_language', 'en'))

            current_store_docs = [doc for doc in current_store.docstore._dict.values()]

            if main_store is None:
                # Initialize with the first valid store
                main_store = current_store
                # Add unique chunks from the first store to the BM25 document list
                for doc in current_store_docs:
                    if doc.page_content not in seen_chunks_for_merge:
                        main_bm25_documents.append(doc.page_content.split())
                        seen_chunks_for_merge.add(doc.page_content)
            else:
                # Merge subsequent stores, avoiding content duplicates
                for doc in current_store_docs:
                    if doc.page_content not in seen_content_for_merge:
                        main_store.add_texts([doc.page_content]) # Add text to FAISS
                        main_bm25_documents.append(doc.page_content.split()) # Add tokenized content for BM25
                        seen_content_for_merge.add(doc.page_content)
        
        if len(indexed_languages) > 1:
            logger.warning(f"Multiple active document sets have different indexed languages ({', '.join(indexed_languages)}). This may affect retrieval accuracy.")

        if main_store and main_bm25_documents:
            # Create a new BM25 index from the combined unique documents
            main_bm25 = BM25Okapi(main_bm25_documents)
            return main_store, main_bm25
        else:
            return None, None

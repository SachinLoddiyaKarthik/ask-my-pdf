import os
import numpy as np # Added numpy import
import logging
import re # Added re import
from typing import List, Dict, Optional, Set

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)

# Define PROMPT_TEMPLATES
PROMPT_TEMPLATES = {
    "summary": """Based on the provided context, summarize the key information related to the question. Be concise and highlight the main points.

    Context:
    {context}

    Question: {question}

    Summary:""",

    "definition": """Using only the provided context, provide a clear and concise definition of the term or concept mentioned in the question.

    Context:
    {context}

    Question: {question}

    Definition:""",

    "data": """Extract and present any relevant data, numbers, or specific facts from the provided context that answer the question. If no direct data is available, state that.

    Context:
    {context}

    Question: {question}

    Data:""",

    "comparison": """Compare and contrast the entities or concepts mentioned in the question, based on the provided context. Highlight similarities and differences.

    Context:
    {context}

    Question: {question}

    Comparison:""",

    "procedure": """Outline the steps or process described in the provided context that are relevant to the question. Present them in a clear, sequential manner.

    Context:
    {context}

    Question: {question}

    Procedure:""",
    
    "contradiction": """Analyze the provided context and the user's statement/question. If there is a contradiction between the context and the user's input, explain it. Otherwise, answer the question based on the context.

    Context:
    {context}

    User Statement/Question: {question}

    Response:""",

    "other": """Answer the question based strictly on the provided context. If the answer is not in the context, state that you cannot answer based on the provided information.

    Context:
    {context}

    Question: {question}

    Answer:"""
}

class LLMHandler:
    def __init__(self): # Removed api_key from __init__ as it's fetched internally
        """
        Initializes the LLMHandler with a Google API key.
        Configures the generative AI model.
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key or len(self.api_key) < 20:
            raise ValueError("Google API Key not found or invalid.")
        genai.configure(api_key=self.api_key)

    def get_embeddings_model(self):
        """Returns a GoogleGenerativeAIEmbeddings model instance."""
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)

    def get_llm(self, temperature: float = 0.3, model_name: str = "gemini-1.5-flash"):
        """Returns a ChatGoogleGenerativeAI LLM instance."""
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    def classify_question(self, query: str) -> str:
        """
        Classifies the question type to guide the LLM's response.
        Uses a zero-shot prompt with a Gemini Flash model.
        """
        prompt = """Classify the following question into one of these categories:
        - summary
        - definition
        - data
        - comparison
        - procedure
        - contradiction
        - other

        Question: {query}
        Category:"""
        
        llm = self.get_llm(temperature=0.0) # Use lower temperature for classification
        response = llm.invoke(prompt.format(query=query)).content
        category = response.strip().lower()
        # Ensure the category is one of the defined prompt templates
        if category not in PROMPT_TEMPLATES:
            return "other"
        return category

    def get_confidence_score(self, source_text: str, answer: str) -> float:
        """
        Calculates a cosine similarity-based confidence score between the generated answer and the source text.
        Embeds both the answer and the source text using the Gemini embedding model and computes their similarity.
        Includes checks for valid embedding results to prevent errors.
        """
        embeddings = self.get_embeddings_model()
        try:
            answer_emb = embeddings.embed_query(answer)
            source_emb = embeddings.embed_query(source_text)

            # Ensure embeddings are valid numerical lists/arrays
            if not isinstance(answer_emb, list) or not answer_emb or \
               not isinstance(source_emb, list) or not source_emb:
                logger.warning("Embedding query returned non-list or empty result for confidence score calculation. Returning 0.0 confidence.")
                return 0.0 # Return 0 confidence if embeddings are invalid or empty

            # Convert to numpy arrays for calculation
            answer_emb_np = np.array(answer_emb)
            source_emb_np = np.array(source_emb)

            # Check for zero norm to prevent division by zero
            if np.linalg.norm(answer_emb_np) == 0 or np.linalg.norm(source_emb_np) == 0:
                logger.warning("Zero norm encountered in embedding for confidence score calculation. Returning 0.0 confidence.")
                return 0.0 

            sim = np.dot(answer_emb_np, source_emb_np) / (np.linalg.norm(answer_emb_np) * np.linalg.norm(source_emb_np))
            return float(sim)
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0

    def get_conversation_chain(self, retriever_instance, memory_instance: ConversationBufferMemory):
        """
        Initializes or retrieves the conversational chain for the active document set(s).
        Uses a custom retriever and a conversation buffer memory.
        """
        llm = self.get_llm()
        
        # Define the prompt for the conversational chain
        qa_prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Answer the question based on the provided context.
            If you cannot find the answer in the context, state that you don't know, but do not make up an answer.
            Keep your answers concise and directly to the point.

            Chat History:
            {chat_history}

            Context:
            {context}

            Question: {question}
            Answer:""",
            input_variables=["chat_history", "context", "question"]
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever_instance,
            memory=memory_instance,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True  # Add for debugging
        )
        return conversation_chain

    def generate_response(self, query: str, retriever_instance, memory_instance: ConversationBufferMemory, detect_language_func, translate_text_func): # Removed target_language as it's not directly used here
        """
        Generates an answer using the conversational chain, RAG fusion, and adds metadata.
        Includes query and answer translation based on detected languages.
        """
        query_lang = detect_language_func(query)

        # Determine the language for retrieval (typically English for embeddings)
        if query_lang != 'en':
            logger.info(f"Translating query from {query_lang} to English for improved retrieval...")
            translated_query_for_retrieval = translate_text_func(query, 'en')
        else:
            translated_query_for_retrieval = query
        
        conversation_chain = self.get_conversation_chain(retriever_instance, memory_instance)
        
        # Classify the original user query for tailored prompting
        qtype = self.classify_question(query)

        try:
            # Run the conversation chain with the translated query
            response_data = conversation_chain.invoke({
                "question": translated_query_for_retrieval
            })
            
            # Explicitly get the answer and source documents
            answer = response_data.get('answer', '')
            retrieved_docs = response_data.get('source_documents', [])
            
            if not answer:
                raise ValueError("No answer generated by the model")
                
            # Translate answer back to user's query language if it was not English
            if query_lang != 'en':
                logger.info(f"Translating answer to {query_lang}...")
                answer = translate_text_func(answer, query_lang)
                
            # Calculate confidence score
            context_for_confidence = "\n\n".join(doc.page_content for doc in retrieved_docs)
            confidence = self.get_confidence_score(context_for_confidence, answer) if context_for_confidence else 0.0

            # Extract sources
            for doc in retrieved_docs:
                match = re.search(r'--- (.*?) - Page (\d+) ---', doc.page_content)
                if match:
                    if 'sources' not in locals(): # Initialize sources if not already
                        sources = set()
                    sources.add(f"{match.group(1)} (Page {match.group(2)})")
            
            return {
                "answer": answer,
                "confidence": float(confidence),
                "sources": sorted(list(sources)) if 'sources' in locals() else [],
                "type": qtype,
                "context_chunks": retrieved_docs[:3] # Top 3 chunks
            }
            
        except Exception as e:
            logger.error(f"Error in conversation chain: {e}")
            return {
                "answer": "Sorry, I encountered an error processing your request. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "type": "error",
                "context_chunks": []
            }

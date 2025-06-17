import streamlit as st
import os
import json
import shutil
import warnings
from datetime import datetime
from typing import List, Dict, Optional, Set

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# Import modules
from llm_handler import LLMHandler
from pdf_utils import PDFUtils, OCR_AVAILABLE, PDF2IMAGE_AVAILABLE
from retriever import Retriever

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Load environment variables
load_dotenv()

# Configure logging (re-configure in UI for Streamlit context if needed)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPDFChatApp:
    def __init__(self):
        """
        Initializes the main application, sets up API keys, and
        creates instances of the handler modules.
        """
        # The API key check is now primarily handled within LLMHandler itself,
        # but a basic check here for early stopping in Streamlit is still useful.
        # The LLMHandler will raise a ValueError if the key is missing/invalid.
        _api_key_check = os.getenv("GOOGLE_API_KEY")
        if not _api_key_check or len(_api_key_check) < 20:
            st.error("❌ Google API Key not found or invalid. Please set GOOGLE_API_KEY in your .env file and ensure it's correct.")
            st.stop()
        
        self.llm_handler = LLMHandler() # Removed api_key argument
        self.pdf_utils = PDFUtils()
        self.retriever_handler = Retriever(llm_handler=self.llm_handler)
        self.last_response = {} # Used for dynamic badge styling
        self.init_session_state()

    def _load_processed_files_info(self):
        """
        Load metadata about previously processed files from FAISS indexes directory.
        This populates the 'Saved Documents' section in the sidebar.
        """
        faiss_root_dir = "faiss_indexes"
        if os.path.exists(faiss_root_dir):
            for store_name in os.listdir(faiss_root_dir):
                store_path = os.path.join(faiss_root_dir, store_name)
                if os.path.isdir(store_path):
                    metadata_path = os.path.join(store_path, "metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            # Fix timestamp handling for backward compatibility
                            if 'timestamp' in metadata:
                                timestamp_value = metadata['timestamp']
                                if isinstance(timestamp_value, str):
                                    try:
                                        metadata['timestamp'] = datetime.fromisoformat(timestamp_value)
                                    except ValueError:
                                        try:
                                            metadata['timestamp'] = datetime.strptime(timestamp_value, '%Y-%m-%d %H:%M:%S.%f')
                                        except ValueError:
                                            try:
                                                metadata['timestamp'] = datetime.strptime(timestamp_value, '%Y-%m-%d %H:%M:%S')
                                            except ValueError:
                                                logger.warning(f"Could not parse timestamp '{timestamp_value}' for {store_name}. Using current time.")
                                                metadata['timestamp'] = datetime.now()
                                elif not isinstance(timestamp_value, datetime):
                                    metadata['timestamp'] = datetime.now()
                            else:
                                metadata['timestamp'] = datetime.now()

                            st.session_state.document_sets[store_name] = metadata
                            
                        except Exception as e:
                            logger.error(f"Error loading metadata for {store_name}: {e}")

    def init_session_state(self):
        """Initializes all necessary session state variables with default values."""
        defaults = {
            'chat_history': [], # Display history (for rendering)
            'document_sets': {}, # Stores metadata for all processed document sets
            'active_sets': set(), # Stores names of currently active document sets
            'current_query': "", # The current query in the input box for suggested follow-ups
            'processing_settings': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'enable_ocr': False,
                'target_language': 'en', # Language for document indexing/translation
                'show_previews': False # Option to show PDF previews
            },
            'current_vector_store_name': None, # Identifier for the currently active *combined* vector store (for chat history linking)
            'chat_memories': {}, # Stores ConversationBufferMemory objects for each document set/combination
            'bm25_retriever_instance': None, # Stores the BM25 instance for the active set (though now managed within Retriever class)
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Load processed files info initially, only if not already loaded
        if not st.session_state.document_sets:
            self._load_processed_files_info()

    def save_chat_history(self, store_id: str):
        """
        Saves chat history for a specific document set combination to a JSON file.
        This provides persistence for conversations linked to specific document sets.
        """
        if not store_id:
            logger.warning("Attempted to save chat history without a valid store_id.")
            return

        store_path = os.path.join("faiss_indexes", store_id)
        os.makedirs(store_path, exist_ok=True) # Ensure directory exists for history
        chat_history_path = os.path.join(store_path, "chat_history.json")
        
        history_data = []
        for msg in st.session_state.chat_history:
            history_data.append({
                "question": msg['question'],
                "answer": msg['answer'],
                "type": msg.get('type'),
                "confidence": msg.get('confidence'),
                "sources": msg.get('sources'),
                "timestamp": msg.get('timestamp')
            })
        
        try:
            with open(chat_history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Chat history for {store_id} saved successfully.")
        except Exception as e:
            logger.error(f"Error saving chat history for {store_id}: {e}")

    def load_chat_history(self, store_id: str):
        """
        Loads chat history for a specific document set combination and rebuilds the
        LangChain ConversationBufferMemory.
        """
        if not store_id:
            st.session_state.chat_history = []
            st.session_state.chat_memories = {} # Clear all memories if no store_id
            return

        store_path = os.path.join("faiss_indexes", store_id)
        chat_history_path = os.path.join(store_path, "chat_history.json")
        
        if os.path.exists(chat_history_path):
            try:
                with open(chat_history_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                st.session_state.chat_history = history_data
                
                # Rebuild ConversationBufferMemory with proper configuration
                memory = ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True,
                    output_key="answer"  # Specify which output to store
                )
                st.session_state.chat_memories[store_id] = memory
                
                for msg in history_data:
                    # Add messages to Langchain memory for conversational context
                    memory.chat_memory.add_user_message(msg['question'])
                    memory.chat_memory.add_ai_message(msg['answer'])
                
                logger.info(f"Chat history for {store_id} loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading chat history for {store_id}: {e}")
                st.session_state.chat_history = [] # Clear if error
                # Still initialize memory even if history load fails
                memory = ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True,
                    output_key="answer"
                )
                st.session_state.chat_memories[store_id] = memory
        else:
            st.session_state.chat_history = []
            # Initialize an empty memory if no history file exists
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
            st.session_state.chat_memories[store_id] = memory

    def render_badge(self, label: str, value: str, icon: str = "🏷️"):
        """
        Theme-aware badge component for displaying metadata like question type, confidence, and sources.
        Dynamically adjusts colors based on Streamlit's theme and the question type.
        """
        theme = "light"  # Default
        try:
            # Attempt to get Streamlit theme base for dynamic styling
            theme = st.get_option("theme.base")
        except:
            pass # Fallback if not running in Streamlit or option unavailable
            
        # Define color scheme for different question types
        colors = {
            "summary": ("#e3f2fd", "#0d47a1"),
            "definition": ("#e8f5e9", "#1b5e20"), 
            "data": ("#fff3e0", "#e65100"),
            "comparison": ("#f3e5f5", "#4a148c"),
            "procedure": ("#e0f7fa", "#006064"),
            "contradiction": ("#ffebee", "#c62828"),
            "other": ("#f5f5f5", "#212121"),
            "error": ("#ffcccc", "#8b0000") # Special color for error
        }
        
        # Use the type from the last response for color selection
        response_type = self.last_response.get('type', 'other')
        bg_light, text_dark_for_light_bg = colors.get(response_type, colors['other'])
        
        # Determine background and text colors based on current theme
        bg_color = bg_light if theme == "light" else text_dark_for_light_bg
        text_color = "#000000" if theme == "light" else "#ffffff"
        border_color = text_dark_for_light_bg if theme == "light" else bg_light # Contrast for border
        
        st.markdown(f"""
        <div style='background-color:{bg_color}; color:{text_color}; padding:6px 12px;
                    border-radius:8px; font-size:0.85em; margin:4px 0; 
                    display:inline-block; border:1px solid {border_color};'>
            {icon} <strong>{label}:</strong> {value}
        </div>
        """, unsafe_allow_html=True)


    def run(self):
        """
        Main method to run the Streamlit application UI.
        Manages the sidebar, file uploads, processing settings, chat interface,
        and displays conversation history.
        """
        st.set_page_config(layout="wide")
        st.title("📚 Multi-PDF Chat Assistant")
        st.markdown("Upload and query across multiple PDF documents with advanced RAG, multilingual support, and more!")

        with st.sidebar:
            st.header("Document Management")
            
            uploaded_files = st.file_uploader(
                "Upload PDFs", 
                type=["pdf"], 
                accept_multiple_files=True,
                help="Upload one or more PDF documents to process."
            )
            
            st.subheader("Processing Settings")
            st.session_state.processing_settings['chunk_size'] = st.slider(
                "Chunk Size", 500, 2000, 
                st.session_state.processing_settings['chunk_size'],
                help="Size of text chunks for processing."
            )
            st.session_state.processing_settings['chunk_overlap'] = st.slider(
                "Chunk Overlap", 50, 500,
                st.session_state.processing_settings['chunk_overlap'],
                help="Overlap between chunks for context preservation."
            )
            st.session_state.processing_settings['enable_ocr'] = st.checkbox(
                "Enable OCR", 
                st.session_state.processing_settings['enable_ocr'],
                disabled=not OCR_AVAILABLE,
                help="Use OCR for scanned/image PDFs (slower). Requires additional installations."
            )
            if not OCR_AVAILABLE:
                st.info("OCR features disabled. Install 'Pillow', 'pytesseract', 'pdf2image' and Tesseract OCR engine.")
            
            st.session_state.processing_settings['target_language'] = st.selectbox(
                "Document Processing Language",
                ["en", "es", "fr", "de", "hi", "zh-CN", "auto"],
                index=["en", "es", "fr", "de", "hi", "zh-CN", "auto"].index(st.session_state.processing_settings['target_language']),
                help="Choose the language for indexing your documents. This helps with multilingual querying."
            )

            st.session_state.processing_settings['show_previews'] = st.checkbox(
                "Show PDF Previews",
                st.session_state.processing_settings['show_previews'],
                disabled=not PDF2IMAGE_AVAILABLE,
                help="Display the first page preview of uploaded PDFs (requires 'pdf2image')."
            )

            if uploaded_files and st.button("Process Documents"):
                with st.spinner("Processing documents... This may take a while for large files."):
                    processed_data = self.pdf_utils.process_documents(uploaded_files, st.session_state.processing_settings)
                    if processed_data:
                        try:
                            store_name = self.retriever_handler.create_vector_store(
                                processed_data, 
                                st.session_state.processing_settings['chunk_size'], 
                                st.session_state.processing_settings['chunk_overlap']
                            )
                            st.session_state.active_sets.add(store_name)
                            st.success(f"Processed {len(uploaded_files)} documents into set: {store_name[:8]}...")
                            
                            if st.session_state.processing_settings['show_previews'] and PDF2IMAGE_AVAILABLE:
                                st.subheader("Uploaded Document Previews")
                                for pdf_file in uploaded_files:
                                    pdf_file.seek(0) # Reset pointer
                                    previews = self.pdf_utils.generate_page_previews(pdf_file.read(), max_pages=1)
                                    pdf_file.seek(0) # Reset pointer again for potential re-reads
                                    if previews:
                                        st.image(previews[0], caption=f"{pdf_file.name} - Page 1 Preview", use_column_width=True)
                                    else:
                                        st.info(f"No preview available for {pdf_file.name}.")
                        except ValueError as ve:
                            st.error(f"⚠️ Processing failed: {ve}")
                            logger.error(f"Processing ValueError: {ve}", exc_info=True)
                        except Exception as e:
                            st.error(f"An unexpected error occurred during processing: {e}")
                            logger.error(f"Unexpected processing error: {e}", exc_info=True)
                    else:
                        st.error("No text could be extracted or processed from the uploaded PDFs.")
            
            st.markdown("---")
            st.subheader("Active Document Sets")
            if st.session_state.document_sets:
                multiselect_options = {}
                for store_id, meta in st.session_state.document_sets.items():
                    file_names = ", ".join([f['name'] for f in meta['files']]) if 'files' in meta and meta['files'] else "Unnamed"
                    timestamp_str = meta['timestamp'] if isinstance(meta['timestamp'], str) else meta['timestamp'].strftime('%Y-%m-%d %H:%M')
                    multiselect_options[store_id] = f"{file_names} (Pages: {meta.get('total_pages', '?')}, Chunks: {meta.get('unique_chunks', '?')}, Lang: {meta.get('indexed_language', 'en')})"
                
                selected_active_sets = st.multiselect(
                    "Select document sets to query:",
                    options=list(st.session_state.document_sets.keys()),
                    default=list(st.session_state.active_sets) if st.session_state.active_sets else [],
                    format_func=lambda x: multiselect_options.get(x, x),
                    key="active_sets_selector"
                )
                
                new_active_sets_hash = "_".join(sorted(selected_active_sets))
                current_active_sets_hash = "_".join(sorted(list(st.session_state.active_sets)))

                if new_active_sets_hash != current_active_sets_hash:
                    # Save history of previous active set combination before switching
                    if st.session_state.current_vector_store_name:
                        self.save_chat_history(st.session_state.current_vector_store_name)
                    
                    st.session_state.active_sets = set(selected_active_sets)
                    st.session_state.current_vector_store_name = new_active_sets_hash
                    self.load_chat_history(st.session_state.current_vector_store_name)
                    st.rerun()

                if st.button("Remove Selected Sets"):
                    if st.session_state.current_vector_store_name in st.session_state.active_sets:
                        self.save_chat_history(st.session_state.current_vector_store_name)
                    
                    for docset_to_remove in selected_active_sets:
                        try:
                            shutil.rmtree(os.path.join("faiss_indexes", docset_to_remove), ignore_errors=True)
                            st.session_state.document_sets.pop(docset_to_remove, None)
                            st.session_state.chat_memories.pop(docset_to_remove, None)
                            st.success(f"Removed document set: {multiselect_options.get(docset_to_remove, docset_to_remove)}")
                        except Exception as e:
                            st.error(f"Error removing set {docset_to_remove}: {e}")
                            logger.error(f"Error deleting FAISS index directory: {e}", exc_info=True)
                    
                    st.session_state.active_sets = st.session_state.active_sets - set(selected_active_sets)
                    st.session_state.current_vector_store_name = "_".join(sorted(list(st.session_state.active_sets))) if st.session_state.active_sets else None
                    self.load_chat_history(st.session_state.current_vector_store_name) # Load history for the new combination
                    st.rerun()

            if st.button("Clear ALL Saved Documents", help="Deletes all processed documents and their chat histories."):
                faiss_root_dir = "faiss_indexes"
                if os.path.exists(faiss_root_dir):
                    try:
                        shutil.rmtree(faiss_root_dir)
                        st.session_state.document_sets = {}
                        st.session_state.active_sets = set()
                        st.session_state.current_vector_store_name = None
                        st.session_state.chat_history = []
                        st.session_state.chat_memories = {}
                        st.success("All saved documents and their chat histories have been deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting all saved documents: {e}")
                        logger.error(f"Error deleting faiss_indexes directory: {e}", exc_info=True)
                else:
                    st.info("No saved documents to delete.")


        # Main Chat Interface
        if st.session_state.active_sets:
            st.subheader("Document Chat")
            
            # Merge vector stores and get the combined BM25 index
            vectorstore, bm25 = self.retriever_handler.merge_vector_stores(st.session_state.active_sets)
            
            if vectorstore:
                st.markdown("### Conversation History")
                
                # Display chat history
                for msg in st.session_state.chat_history:
                    if msg.get('question'):
                        with st.chat_message("user"):
                            st.markdown(f"**You:** {msg['question']}")
                    if msg.get('answer'):
                        with st.chat_message("assistant"):
                            self.last_response = msg # Set last_response for badge coloring
                            st.markdown(msg['answer'])
                            if msg.get('type'):
                                self.render_badge("Type", msg['type'], "🔖")
                            if msg.get('confidence') is not None:
                                self.render_badge("Confidence", f"{msg['confidence']:.2f}", "📊")
                            if msg.get('sources'):
                                self.render_badge("Sources", ", ".join(msg['sources']), "📄")
                
                # Use a form to handle query submission to clear input after submission
                with st.form(key="query_form"):
                    query = st.text_input(
                        "Ask a question about the documents:",
                        key="query_input",
                        value=st.session_state.current_query # Pre-fill from suggested follow-ups
                    )
                    submit_button = st.form_submit_button("Submit")

                if submit_button and query:
                    st.session_state.current_query = "" # Clear pre-filled query after submission
                    with st.spinner("Analyzing documents..."):
                        try:
                            # Instantiate custom retriever for the current query
                            retriever_instance = self.retriever_handler.CustomRagFusionRetriever(
                                vectorstore=vectorstore,
                                bm25=bm25,
                                embeddings_model=self.llm_handler.get_embeddings_model(),
                                parent_retriever=self.retriever_handler # Pass self reference
                            )
                            
                            # Get the appropriate chat memory for the current active set combination
                            active_set_id = "_".join(sorted(list(st.session_state.active_sets)))
                            if active_set_id not in st.session_state.chat_memories:
                                self.load_chat_history(active_set_id) # Ensure memory is loaded
                            memory = st.session_state.chat_memories[active_set_id]

                            response = self.llm_handler.generate_response(
                                query, 
                                retriever_instance, 
                                memory,
                                self.pdf_utils._detect_language, # Pass PDFUtils methods for language handling
                                self.pdf_utils._translate_text
                                # Removed target_language as it's not directly used by generate_response
                            )
                            self.last_response = response # Update for badge rendering
                            
                            # Append the new interaction to the session chat history
                            st.session_state.chat_history.append({
                                'question': query,
                                'answer': response['answer'],
                                'type': response['type'],
                                'confidence': response['confidence'],
                                'sources': response['sources'],
                                'timestamp': datetime.now().isoformat(),
                                'context_chunks': response.get('context_chunks', []) # Store context chunks for display
                            })
                            
                            # Save chat history after each interaction
                            self.save_chat_history(active_set_id)

                            # Rerun to clear the input box and update chat history display
                            st.rerun()

                        except Exception as e:
                            st.error(f"An error occurred while answering: {e}")
                            logger.error(f"Answer generation error: {e}", exc_info=True)
                
                # Suggested Follow-ups
                if st.session_state.chat_history:
                    with st.expander("💡 Suggested Follow-ups"):
                        cols = st.columns(3)
                        last_question = st.session_state.chat_history[-1]['question'] if st.session_state.chat_history else ""

                        if last_question:
                            with cols[0]:
                                if st.button("Explain more", key=f"explain_{len(st.session_state.chat_history)}"):
                                    st.session_state.current_query = f"Explain in more detail: {last_question}"
                                    st.rerun()
                            with cols[1]:
                                if st.button("Get examples", key=f"examples_{len(st.session_state.chat_history)}"):
                                    st.session_state.current_query = f"Provide examples about: {last_question}"
                                    st.rerun()
                            with cols[2]:
                                if st.button("Summarize", key=f"summarize_{len(st.session_state.chat_history)}"):
                                    st.session_state.current_query = f"Summarize key points about: {last_question}"
                                    st.rerun()
                
                    # View Relevant Document Sections (from the last response)
                    if st.session_state.chat_history[-1].get('context_chunks'):
                        with st.expander("🔍 View Relevant Document Sections"):
                            for i, chunk in enumerate(st.session_state.chat_history[-1]['context_chunks']):
                                st.markdown(f"**Relevant Section {i+1}**")
                                st.text_area("", chunk.page_content, height=150, key=f"chunk_display_{len(st.session_state.chat_history)}_{i}")
                    
                    st.markdown("---")
                    # Export Chat History button
                    chat_export_data = []
                    for item in st.session_state.chat_history:
                        chat_export_data.append({
                            "role": "user" if "question" in item else "assistant",
                            "content": item.get('question') if "question" in item else item.get('answer'),
                            "type": item.get('type'),
                            "confidence": item.get('confidence'),
                            "sources": item.get('sources'),
                            "timestamp": item.get('timestamp')
                        })
                    
                    json_string = json.dumps(chat_export_data, indent=2, ensure_ascii=False)
                    current_active_set_id_display = "_".join(sorted(list(st.session_state.active_sets)))
                    st.download_button(
                        label="Export Chat History (.json)",
                        data=json_string,
                        file_name=f"chat_history_{current_active_set_id_display}.json",
                        mime="application/json",
                        key="download_chat_history"
                    )

                # Clear Current Chat button
                if st.button("Clear Current Chat", help="Clears the displayed conversation for the active document sets."):
                    st.session_state.chat_history = []
                    current_active_set_id = "_".join(sorted(list(st.session_state.active_sets)))
                    if current_active_set_id in st.session_state.chat_memories:
                        st.session_state.chat_memories[current_active_set_id].clear() # Clear LangChain memory
                    
                    self.save_chat_history(current_active_set_id) # Save empty history
                    st.success("Chat history cleared.")
                    st.rerun()

            else:
                st.warning("No valid document sets loaded for querying. Please select or process documents.")
        else:
            st.info("Please upload and process documents or select existing document sets from the sidebar to begin chatting.")
            st.markdown("""
                ### Welcome to Chat with your PDF!
                This application allows you to interactively chat with your PDF documents.
                Upload a PDF, ask questions, and get intelligent answers with context from your document.

                **Key Features:**
                - **Gemini Flash Integration:** Powered by Google's Gemini Flash model.
                - **Advanced RAG (RRF):** Combines semantic and keyword search for accurate answers.
                - **Multilingual Support:** Translate documents during indexing and queries for conversation.
                - **Query Classification:** Answers tailored to question type (summary, definition, data, etc.).
                - **Confidence Scores:** Shows answer reliability.
                - **Source Citations:** Shows document sections used.
                - **OCR Support:** Works with scanned PDFs (requires additional installations).
                - **Persistent Storage:** Saves processed documents and their conversation histories.
                - **Session Management:** Maintains conversation history per active document set(s).
                - **Multi-Document Chat:** Query across multiple selected PDF documents simultaneously.

                **To get started:**
                1. Upload a PDF using the file uploader in the sidebar.
                2. Adjust processing settings (Chunk Size, Overlap, OCR, Language).
                3. Click "Process Documents" and wait for completion.
                4. Or, select a previously processed document set under "Saved Documents".
                5. Start asking questions in the "Document Chat" section!
                
                _Made with ❤️ for your documents._
            """)

# Ensure the faiss_indexes directory exists on startup
if __name__ == "__main__":
    os.makedirs("faiss_indexes", exist_ok=True)
    app = EnhancedPDFChatApp()
    app.run()

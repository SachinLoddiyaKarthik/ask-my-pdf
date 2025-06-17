import os
import io
import re
import hashlib
import time
import logging
from typing import List, Dict, Optional # Added Optional
from datetime import datetime

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator, single_detection

# Optional imports for OCR and PDF preview
try:
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    PDF2IMAGE_AVAILABLE = False
    # Streamlit warnings will be handled in the UI layer

# Configure logging
logger = logging.getLogger(__name__)

class PDFUtils:
    def __init__(self):
        pass

    def _translate_text(self, text: str, dest_lang: str) -> str:
        """
        Translates text to the destination language using Google Translator.
        Avoids translation if the text is empty, destination is 'auto', or
        the text is already in the destination language.
        """
        if not text or dest_lang == 'auto' or dest_lang == self._detect_language(text):
            return text
        try:
            return GoogleTranslator(source='auto', target=dest_lang).translate(text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text

    def _detect_language(self, text: str) -> str:
        """
        Detects the language of the given text using DeepL's single_detection (if API key available).
        Falls back to simpler heuristics or 'en' if DeepL API key is not found or detection fails.
        """
        try:
            api_key = os.getenv("DEEPL_API_KEY")
            if api_key:
                detected_lang = single_detection(text, api_key=api_key)
                return detected_lang if detected_lang else "en"
            else:
                logger.warning("DEEPL_API_KEY not found. Language detection might be less accurate.")
                # Simple heuristic for common languages
                if re.search(r'[\u0400-\u04FF]', text): return 'ru' # Cyrillic
                if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text): return 'ja' # Japanese/Chinese
                return 'en' # Default fallback
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
            return 'en'

    def _index_document_multilingual(self, text: str, target_language: str) -> str:
        """
        Translates the document text to a target language for indexing if necessary.
        This ensures that documents are indexed in a consistent language,
        aiding in multilingual querying.
        """
        if target_language == 'en' or not text:
            return text
        current_lang = self._detect_language(text)
        if current_lang != target_language:
            logger.info(f"Translating document content from {current_lang} to {target_language} for indexing...")
            return self._translate_text(text, target_language)
        return text

    def clean_text(self, text: str) -> str:
        """
        Normalize and clean text content by removing control characters and extra whitespace.
        This helps in processing clean and consistent text.
        """
        cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
        cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text) # Remove control characters
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # Replace multiple spaces with single space
        return cleaned_text

    def _extract_page_text(self, page, pdf_bytes: bytes, page_number: int, enable_ocr: bool) -> str:
        """
        Extract text from a PDF page with optional OCR fallback.
        If direct text extraction fails and OCR is enabled and available,
        it attempts to use Tesseract OCR on the page image.
        """
        text = ""
        try:
            text = page.extract_text()
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_number + 1}: {e}")

        if not text and enable_ocr and OCR_AVAILABLE:
            if PDF2IMAGE_AVAILABLE:
                try:
                    # convert_from_bytes expects 1-indexed page numbers
                    images = convert_from_bytes(pdf_bytes, first_page=page_number + 1, last_page=page_number + 1)
                    if images:
                        text = pytesseract.image_to_string(images[0])
                except Exception as e:
                    logger.error(f"OCR error for page {page_number + 1}: {e}")
        return self.clean_text(text) if text else ""

    def process_documents(self, pdf_files: List, processing_settings: Dict) -> Optional[Dict]:
        """
        Process multiple PDFs into a unified text corpus with metadata.
        Iterates through each PDF, extracts text page by page (with optional OCR),
        and combines it into a single string.
        """
        if not pdf_files:
            return None

        combined_text = ""
        metadata = {
            'files': [],
            'total_pages': 0,
            'total_chars': 0,
            'timestamp': datetime.now().isoformat(),
            'file_hashes': [],
            'indexed_language': processing_settings['target_language'] # Default language
        }
        
        start_time = time.time()

        for pdf in pdf_files:
            try:
                pdf_bytes = pdf.read() # Read bytes once
                file_hash = hashlib.sha256(pdf_bytes).hexdigest()
                pdf.seek(0) # Reset pointer for PdfReader if it was read

                reader = PdfReader(io.BytesIO(pdf_bytes)) # Create PdfReader from bytes
                file_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    page_text = self._extract_page_text(page, pdf_bytes, page_num, processing_settings['enable_ocr'])
                    if page_text:
                        # Add a separator to distinguish content from different pages/files in the combined text
                        file_text += f"\n--- {pdf.name} - Page {page_num+1} ---\n{page_text}\n"
                
                if file_text:
                    combined_text += file_text
                    metadata['files'].append({
                        'name': pdf.name,
                        'pages': len(reader.pages),
                        'chars': len(file_text),
                        'hash': file_hash
                    })
                    metadata['total_pages'] += len(reader.pages)
                    metadata['total_chars'] += len(file_text)
                    metadata['file_hashes'].append(file_hash)
                    
            except Exception as e:
                logger.error(f"Error processing {pdf.name}: {e}")
                continue

        if not combined_text.strip():
            logger.error("No readable text extracted from PDFs.")
            return None

        # Translate combined text for indexing if target language is not English
        if metadata['indexed_language'] != 'en':
            combined_text = self._index_document_multilingual(combined_text, metadata['indexed_language'])
            
        metadata['processing_time'] = time.time() - start_time
        return {'text': combined_text, 'metadata': metadata}

    def create_text_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Splits the text into chunks using RecursiveCharacterTextSplitter.
        Filters out very small chunks to ensure meaningful content per chunk.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100] # Filter out very small chunks
        return chunks

    def generate_page_previews(self, pdf_file_content: bytes, max_pages=1):
        """
        Generates image previews for a PDF file from its bytes.
        Requires PDF2IMAGE_AVAILABLE to be True.
        """
        if not PDF2IMAGE_AVAILABLE:
            return []

        try:
            images = convert_from_bytes(pdf_file_content)
            
            previews = []
            for i, image in enumerate(images[:max_pages]):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                previews.append(img_byte_arr.getvalue())
            return previews
        except Exception as e:
            logger.error(f"Error generating PDF previews: {e}")
            return []

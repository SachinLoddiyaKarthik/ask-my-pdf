# 📄 Ask My PDF Pro — Chat with Multiple PDFs using LLMs, FAISS & BM25

An intelligent, multilingual PDF assistant that lets you **upload, chat with, and extract answers** from multiple documents using **Hybrid Retrieval (Vector + BM25)** and **LLMs (Gemini / ChatGPT)**. From research papers and invoices to contracts and policy reports — this app handles them all with contextual understanding.

### 📦 Live Demo

🚀 Check out the deployed app here:  
👉 **[ask-my-pdf-pro.streamlit.app](https://ask-my-pdf-pro.streamlit.app/)**

[![Streamlit App](https://img.shields.io/badge/Live%20App-Ask%20My%20PDF%20Pro-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ask-my-pdf-pro.streamlit.app/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-00A896?style=for-the-badge)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/BM25-Retrieval-orange?style=for-the-badge)](https://en.wikipedia.org/wiki/Okapi_BM25)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

---

## 🔥 Features

- 📁 **Upload & Chat with Multiple PDFs**
- 🧠 **Hybrid Retrieval**: FAISS + BM25
- 🌐 **Multilingual Support** with Translation & Detection
- 🧾 **Citation-Aware Answers** with Source Highlighting
- 🗂 **Document Set Management** with Chat Memory per PDF
- 📤 **Export Chat History** (Coming soon)
- 👁️ **OCR Support** (Optional with `pytesseract`)

---

## 🧠 How It Works

1. **PDF Upload** → Extract and chunk content using PyMuPDF.
2. **Hybrid RAG Retrieval** → FAISS for semantic search + BM25 for keyword match.
3. **LLM-based Response** → Answer questions using Gemini Pro / OpenAI via LangChain.
4. **Context Tracking** → Store chat history per document set.

---

## 🛠️ Tech Stack

| Component        | Description                            |
|------------------|----------------------------------------|
| `Streamlit`      | Interactive web UI                     |
| `LangChain`      | Prompting and retrieval abstraction    |
| `FAISS`          | Embedding-based document search        |
| `BM25`           | Keyword-based fallback retriever       |
| `PyMuPDF`        | Fast text & image extraction from PDFs |
| `Google Gemini`  | Main LLM for QA                        |
| `dotenv`         | API Key handling                       |

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/SachinLoddiyaKarthik/ask-my-pdf-pro.git
cd ask-my-pdf-pro
````

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Your API Keys

Create a `.env` file like below:

```env
GOOGLE_API_KEY=your_google_gemini_key
DEEPL_API_KEY=your_deepl_key_if_any
```

---

## 🎯 Usage

```bash
streamlit run ui.py
```

Then go to `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
ask-my-pdf-pro/
├── llm_handler.py       # Handles LLM interactions (Gemini, LangChain)
├── pdf_utils.py         # PDF reading, chunking, translation, OCR
├── retriever.py         # Hybrid Retriever (FAISS + BM25)
├── ui.py                # Streamlit UI logic
├── requirements.txt     # Project dependencies
├── .env.example         # Example of environment variables
└── README.md            # You’re reading it!
```

---

## 🧪 Example Prompts

* "Summarize the key findings in this document."
* "What are the listed risks in section 3.1?"
* "Translate the paragraph about tax rates into Spanish."
* "Who are the stakeholders mentioned in the appendix?"

---

## ⚠️ Limitations

| Feature      | Limitation                               |
| ------------ | ---------------------------------------- |
| OCR          | Requires Tesseract + PIL manually        |
| LLM Accuracy | Depends on chunking + retrieval quality  |
| GPU FAISS    | Not supported by default (CPU mode used) |

---

## 🌟 Future Plans

* ✅ Streamlit Chat UI with expandable memory
* 🔄 Chat Export (CSV, Markdown)
* 🗃️ Multi-PDF Set Comparison
* 📸 Live Preview + Snippet Highlighting
* 🌐 UI Internationalization

---

## 🤝 Contribute

Pull requests and issues are welcome!
Feel free to fork and extend — contributions from the community are appreciated.

---

## 👨‍💻 Author

**Sachin Loddiya Karthik**
🔗 [LinkedIn](https://www.linkedin.com/in/sachin-lk/)
💻 [GitHub](https://github.com/SachinLoddiyaKarthik)

---

## 🌈 Like This Project?

Give it a ⭐ on GitHub! It keeps the motivation flowing!

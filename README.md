
# 📄 Ask My PDF 

An intelligent question-answering system that lets you ask questions from PDF documents like government reports, budget speeches, or research papers using **LangChain**, **OpenAI**, and **FAISS**, all in a simple notebook workflow.


[![LangChain](https://img.shields.io/badge/LangChain-00A896?style=for-the-badge)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

---

## 🎯 Project Overview

This project enables conversational interaction with the content of a PDF document using language models. It leverages:

- 📄 **PyMuPDF** to parse PDF content
- 🧠 **LangChain** to structure the Q&A logic
- 🔎 **FAISS** for semantic search
- 🤖 **OpenAI** for language model responses

The result is a lightweight, no-UI notebook system to explore documents intelligently.

---

## 🛠 Tech Stack

| Tool/Library   | Purpose                                 |
|----------------|------------------------------------------|
| Python         | Base programming language               |
| LangChain      | Framework for chaining LLM tasks        |
| OpenAI         | LLM provider for intelligent responses  |
| FAISS          | Vector database for similarity search   |
| PyMuPDF (fitz) | PDF parsing and text extraction         |
| Jupyter        | Interactive development and exploration |
| dotenv         | Secure management of API credentials    |

---

## 🧪 How It Works

1. 📥 Load and parse a PDF file using `PyMuPDF`
2. ✂️ Chunk the text and generate vector embeddings
3. 📚 Store embeddings in a FAISS vector database
4. ❓ Accept user questions
5. 🔍 Retrieve relevant chunks using semantic similarity
6. 💬 Use OpenAI to generate a response based on retrieved context

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/ask-my-pdf.git](https://github.com/SachinLoddiyaKarthik/ask-my-pdf.git)
cd ask-my-pdf
````

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

Create a `.env` file and add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Launch the Notebook

Open the notebook in Jupyter:

```bash
jupyter notebook pdfQuery.ipynb
```

---

## ✨ Example Questions You Can Ask

* "What is the fiscal deficit announced in the speech?"
* "Summarize the MSME-related budget policies."
* "What are the new tax changes mentioned in this document?"

---

## 📄 File Structure

```
📁 ask-my-pdf/
├── pdfQuery.ipynb         # Main notebook for Q&A
├── budget_speech.pdf      # Sample input PDF
├── requirements.txt       # Required dependencies
├── .env.example           # Sample env file
└── README.md              # Project documentation
```

---

## 🧠 Limitations

* Works best with **well-formatted textual PDFs**
* Not yet integrated into a Streamlit or web interface
* Performance depends on LLM availability and API key usage

---

## 📬 Contact

* **GitHub**: [SachinLoddiyaKarthik](https://github.com/SachinLoddiyaKarthik)
* **LinkedIn**: [Connect on LinkedIn](https://www.linkedin.com/in/sachin-lk/)

---

⭐ Star this repo if it helped you explore documents with LLMs!

```


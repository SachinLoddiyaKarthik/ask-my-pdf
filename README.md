
# 📄 Ask My PDF – Chatbot with LangChain & Streamlit

[![LangChain](https://img.shields.io/badge/LangChain-00A896?style=for-the-badge)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FCC624?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

## 🎯 Project Overview

An interactive chatbot app that allows users to ask questions about PDF documents—especially government or financial PDFs like Budget Speeches—using LangChain, OpenAI, and Streamlit.

### 💡 Highlights

- 📄 **PDF ingestion and chunking** using LangChain's `PyPDFLoader`
- 🔗 **LangChain RetrievalQA** for context-aware document querying
- 🤖 **OpenAI API** for generating intelligent responses
- 🧠 **FAISS** for efficient similarity search over text chunks
- 🔐 **dotenv integration** to securely manage API keys
- 📺 **Streamlit interface** for user-friendly interaction

## 🏗️ Architecture

```
PDF → Text Chunking → Embeddings (FAISS) → LangChain Retrieval → OpenAI LLM → Response
```

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SachinLoddiyaKarthik/ask-my-pdf.git
cd ask-my-pdf
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure `.env` File

Create a `.env` file and add your API key:

```env
OPENAI_API_KEY=your_openai_key_here
```

### 5. Run the App

```bash
streamlit run app.py
```

## 🛠 Tech Stack

| Tool         | Usage                                      |
|--------------|---------------------------------------------|
| Python       | Programming language                        |
| Streamlit    | Frontend interface                          |
| LangChain    | LLM chaining and document processing        |
| OpenAI       | Large language model for answer generation  |
| FAISS        | Semantic similarity search across chunks    |
| dotenv       | Secure environment configuration            |
| PyMuPDF      | PDF parsing and content extraction          |

## ✨ Sample Output

```
User: What is the fiscal deficit for 2025-26?
Bot: The fiscal deficit for 2025-26 is estimated at 5.1% of GDP...
```

## 🤝 Contributing

Feel free to fork, improve, and submit a pull request. Star ⭐ the repo if it helps you!

## 📞 Contact

- **GitHub**: [SachinLoddiyaKarthik](https://github.com/SachinLoddiyaKarthik)
- **LinkedIn**: [Connect with me](https://www.linkedin.com/in/sachin-lk/)

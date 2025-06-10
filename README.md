# 📄 Ask My PDF: From Precise Scraping to Intelligent Q\&A

A hybrid toolkit that blends the **surgical precision of layout-based PDF extraction** with the **intelligence of LLM-powered question answering**. Whether you're parsing invoice tables or conversing with a government budget report, this project has your back.

[![LangChain](https://img.shields.io/badge/LangChain-00A896?style=for-the-badge)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge\&logo=openai\&logoColor=white)](https://openai.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org/)

---

## 🎯 Project Modes

### 🧱 1. Layout-Based Extraction (`PDFQuery.ipynb`)

* Uses `pdfquery` to dig into text at specific positions or with specific styling.
* Perfect for **structured PDF documents** like forms, invoices, or certificates.
* Think: "Give me the text at coordinates (150, 500)!"

### 🧠 2. LLM-Powered Conversational Q\&A (`pdfQuery.ipynb`)

* Combines **LangChain**, **OpenAI**, **FAISS**, and **PyMuPDF**.
* Great for **long-form, unstructured documents** like budget speeches or research papers.
* Think: “Summarize the tax changes mentioned in this document.”

---

## 🧰 Tech Stack

| Component    | Used In     | Purpose                             |
| ------------ | ----------- | ----------------------------------- |
| `pdfquery`   | Layout Mode | XML-style querying of text layout   |
| `PyMuPDF`    | Q\&A Mode   | Extracts raw text from PDFs         |
| `FAISS`      | Q\&A Mode   | Embedding-based chunk retrieval     |
| `LangChain`  | Q\&A Mode   | Manages prompts, chaining, logic    |
| `OpenAI API` | Q\&A Mode   | LLM for natural language answers    |
| `dotenv`     | Q\&A Mode   | Secure API credential management    |
| `Jupyter`    | Both        | Interactive, step-by-step notebooks |

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/SachinLoddiyaKarthik/ask-my-pdf.git
cd ask-my-pdf
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file and add:

```env
OPENAI_API_KEY=your_openai_key_here
```

---

## 💡 How to Use

### 📊 Precise Layout-Based Extraction

Open `PDFQuery.ipynb` in Jupyter Notebook to:

* Locate and extract text from exact positions
* Query based on fonts, sizes, and coordinates

### 💬 Ask My PDF: Chat with Your Docs

Open `pdfQuery.ipynb` to:

* Chunk and vectorize your PDF
* Ask natural language questions
* Get AI-generated answers with contextual snippets

---


## 📋 Sample Questions for `budget_speech.pdf`

These are intelligent, real-world prompts users can try out in your app or notebook:

### 🧠 General Summary

* "Summarize the key themes of Budget 2025-26."
* "What are the main objectives of this year’s budget?"
* "What is meant by the term ‘Viksit Bharat’ in this budget?"

### 🌾 Agriculture & Rural

* "What is the Prime Minister Dhan-Dhaanya Krishi Yojana?"
* "How does the budget plan to boost rural prosperity?"
* "What is the Mission for Aatmanirbharta in Pulses?"
* "Explain the focus of the Rural Prosperity and Resilience programme."
* "What is the government doing to improve cotton productivity?"

### 🏢 MSMEs and Startups

* "What reforms were announced for MSMEs?"
* "How will credit availability be improved for small businesses?"
* "What are the features of the new scheme for first-time entrepreneurs?"
* "What support is provided for the toy and leather sectors?"

### 🧬 Innovation & Research

* "What investments are proposed in research and innovation?"
* "Tell me about the PM Research Fellowship scheme."
* "What is the Deep Tech Fund of Funds?"

### 👩‍🏫 Education & Health

* "What are the plans to enhance medical education in India?"
* "How will the government support school students and language learning?"
* "What initiatives are planned for skilling youth?"
* "What healthcare benefits are planned for gig workers?"

### 🚰 Infrastructure & Urban Development

* "What are the new PPP infrastructure projects announced?"
* "Tell me about the Urban Challenge Fund."
* "What is the plan for the Western Koshi Canal in Bihar?"

### 🌐 Exports & Industry

* "What are the four engines of economic growth mentioned?"
* "How will BharatTradeNet support international trade?"
* "Which sectors will be supported to integrate with global supply chains?"

### 💰 Tax & Fiscal

* "What are the new personal income tax slabs?"
* "What changes have been made to TDS and TCS rules?"
* "What is the fiscal deficit target for 2025-26?"

### 🌍 Energy & Environment

* "What is the Nuclear Energy Mission for Viksit Bharat?"
* "What measures are proposed for clean tech manufacturing?"

---

## 📁 Folder Structure

```
ask-my-pdf/
├── PDFQuery.ipynb         # Layout-based extraction notebook
├── pdfQuery.ipynb         # LLM Q&A over PDF
├── requirements.txt       # Project dependencies
├── .env.example           # Sample environment file
├── sample.pdf             # Add your PDFs here
└── README.md              # This file!
```

---

## ⚠️ Limitations

| Feature             | Limitation                           |
| ------------------- | ------------------------------------ |
| `pdfquery`          | Doesn’t work on image-based PDFs     |
| `Ask My PDF` (Q\&A) | Relies on OpenAI API & good chunking |
| Both                | No web UI (yet 😉)                   |

---

## 🤝 Contribute & Explore

Feel free to fork, star ⭐, or submit PRs for improvements. Future ideas:

* Streamlit UI
* OCR support
* Multi-PDF chaining

---

## 📬 Connect with Me

**Author**: [Sachin Loddiya Karthik](https://www.linkedin.com/in/sachin-lk/)
**GitHub**: [SachinLoddiyaKarthik](https://github.com/SachinLoddiyaKarthik)

---

## 🌟 Like It?

Star this repo if it saved you hours of manual PDF scraping or searching. Your support motivates more such tools! 🙌

---

Would you like me to now:

* 📦 Generate the `requirements.txt` based on both notebooks
* 🧾 Create the `.env.example` file

So the repo is fully launch-ready?

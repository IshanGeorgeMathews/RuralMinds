# RuralMinds

**RuralMinds** is a decentralized academic infrastructure engineered to bridge the educational divide where connectivity fails but learning must continue. 

## Problem Statement

Education in India follows a unified academic framework, yet the quality of delivery varies drastically between urban institutions and rural or internet-restricted regions. While urban students benefit from instant access to online resources, AI tools, and collaborative platforms, rural learners often face systemic barriers that directly affect learning outcomes. 

In many rural schools:
- **Teacher shortages** force a single educator to handle multiple subjects across different grades, reducing preparation quality and subject depth. 
- **Learning discontinuity** occurs during teacher absences, leaving classrooms unguided.
- **Limited, inconsistent, or expensive internet access** makes reliance on cloud-based platforms impractical.
- **Language barriers** complicate instruction, especially when teachers are deployed to regions where they are not fluent in the local language. 
- **No updated study materials** or peer collaboration opportunities for rural teachers.

At a systemic level, solving connectivity through full-scale internet infrastructure introduces multiple failure points—ISP dependency, power instability, and high maintenance requirements. Establishing a government-supported internet tower can cost between ₹7.5–19.5 lakhs initially, with recurring annual expenses reaching ₹2.3–6.3 lakhs, and a 10-year total cost potentially exceeding ₹30–82 lakhs.

## The RuralMinds Solution

RuralMinds proposes a cost-optimized, **offline-first academic infrastructure** designed specifically for rural and restricted-internet environments. Instead of relying on continuous internet access, the system operates through a local intranet server model using a low-power single-board computer (like a Raspberry Pi). 

With a setup cost between ₹21,000–63,000 per node and minimal annual maintenance, this approach enables schools or villages to deploy a high-speed LAN-based educational ecosystem at a fraction of traditional infrastructure costs.

---

## System Architecture & Dual-DBMS

To maintain resilience and robust data integrity in off-grid environments, RuralMinds is built on a custom **Dual-Database Management System (DBMS)** architecture. This separates structured relational data from unstructured AI semantic data.

### 1. Relational Database Engine (SQLite)
The backbone of the application state relies on a highly optimized, standard C-library SQL Database (SQLite). 
*   **ACID Compliant:** Transactions are strictly Atomic, Consistent, Isolated, and Durable. In regions with unstable power, unexpected shutdowns will not corrupt user credentials or forum states.
*   **Zero-Configuration:** It requires no background database servers (unlike PostgreSQL), making it exceptionally rugged for immediate deployment.
*   **Data Models:** It strictly tracks normalized schemas for `users` (Authentication & Security), `forum_posts` (Discussion Contexts), `forum_replies` (Threaded mappings connected via Foreign Keys), and `documents` (System Audit tracking).

### 2. Semantic Vector Engine (ChromaDB)
For AI search capabilities, standard keyword SQL searches are insufficient.
*   **Document Intelligence:** When teachers upload PDFs, the documents are processed and split into numeric vectors (embeddings).
*   **Semantic RAG:** ChromaDB operates as a highly specialized local vector database predicting the logical "closeness" of a student's question to the text inside an uploaded PDF, fetching exactly what the Local Mistral LLM needs to answer the question perfectly without the internet.

### 3. Persistent Binary Storage
Beyond just vectorizing PDFs, RuralMinds securely archives original binary documents inside a secure `/uploaded_pdfs/` vault. The SQLite database retains exact path references so that students and teachers can natively Download and View original textbook PDFs directly within the application.
It also fully integrates robust Video Storage alongside generated or user-provided subtitles (saved securely in the `/captions/` directory), establishing an offline streaming capability for educational accessibility.

---

## Core Platform Views

### 1. Student AI Learning Assistant (Offline LLM-Based Q&A)
- Students can ask doubts in their native script or in romanized form, and the multilingual model responds appropriately. This ensures accessibility even for learners uncomfortable with English. 
- The system also helps students refine answers into well-structured academic English, particularly useful during board examinations. 
- Unlike unrestricted internet AI, the assistant operates strictly within curated teacher-provided materials logged in ChromaDB, reducing hallucination risks and ensuring syllabus alignment.

### 2. Discussion Forum with Structured Moderation
- Fully backed by SQLite SQL queries, students can interact, discuss concepts, and upvote relevant doubts—prioritizing common academic concerns similar to structured community platforms. 
- Teachers maintain full moderation control, ensuring disciplined and focused discussions. 

### 3. Teacher Content Publishing & Smart Document Processing
- Teachers can upload lecture videos and PDFs directly to the local database without internet.
- Intelligent document processing directly integrates OCR. Because many educational PDFs are image-based and inefficient for standard language models, RuralMinds extracts text from scanned PDFs and converts them into structured ChromaDB datasets. 

---

## System Security

To maintain institutional integrity, the system includes a **role-based authentication architecture** hardened by SQLite security queries:
- **Student registration** remains open within defined parameters.
- **Teacher registration** requires administrative authorization. 
- An **admin control layer** enables verification, addition, removal, and monitoring of educators, ensuring only certified individuals access instructional privileges. passwords are fundamentally secured using SHA-256 cryptographic hashes.

## Technology Stack

**Frontend:** Streamlit  
**Master Relational DB:** SQLite3 (Core DBMS, ACID)  
**Vector DB / Semantic Engine:** ChromaDB  
**Local Generative AI:** Ollama running Mistral LLM  
**Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)  
**Text Processing & OCR:** PyMuPDF, `pdf2image`, `pytesseract`  
**Audio & Language:** OpenAI Whisper, Hugging Face Transformers  
**Language:** Python 3.9+  

---

## Code Purpose & How to Run

### Purpose
The RuralMinds codebase creates a local web application acting as the intranet node software. It facilitates smart PDF vectorization, offline RAG querying, audio transcription, multilingual translation, and relational community forum management. By utilizing a hybrid dual-DBMS approach, it dramatically reduces deployment costs and completely eliminates internet dependency.

### Prerequisites
- Python 3.9+
- pip (Python package installer)
- Git
- *Tesseract OCR* engine installed on systemic level (required for `pytesseract`)
- *Poppler* tools installed on systemic level (required for `pdf2image`)
- *Ollama* installed and running locally with the `mistral` model downloaded (`ollama run mistral`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/RuralMinds.git
   cd RuralMinds
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### First Run & Automatic Migration
Upon the first initialization, RuralMinds will systematically establish the `ruralminds.db` SQLite database schema. If legacy `users_db.json` or `forum_db.json` data files are present, the system will seamlessly migrate them into the new ACID-compliant SQLite structure and preserve the originals as `.bak` files.

### Configuration

Default administrator credentials (configured securely upon initial database generation):
- **Username:** `admin`
- **Password:** `administrator`
*(Important: Use these to login and create Teacher accounts, then change password)*

### Running the Application

Start the application on your local server:
```bash
streamlit run app.py
```
Access the application by opening a browser and navigating to:
`http://localhost:8501` or your intranet IP address `http://<your-ip>:8501`.
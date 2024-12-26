# Gemini PDF Chatbot

This project is a PDF-based chatbot powered by Google Generative AI (Gemini-Pro) and LangChain. It enables users to upload PDF files, process their content into searchable chunks, and interact with the information through natural language queries. The chatbot leverages embeddings and conversational chains for precise and context-aware responses.

## Features

- **PDF Upload and Processing:** Upload multiple PDF files, extract text, and preprocess into chunks for embedding.
- **Semantic Search:** Search for answers based on similarity search using FAISS vector store.
- **Conversational Interface:** Engage with the PDF content through a chatbot interface powered by Streamlit.
- **Google Generative AI Integration:** Use Gemini-Pro for embedding generation and question-answering tasks.

## Prerequisites

- Python 3.8+
- Google API Key for Generative AI
- Required Python libraries:
  - `google-generativeai`
  - `streamlit`
  - `PyPDF2`
  - `langchain`
  - `langchain-google-genai`
  - `python-dotenv`

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   - Create a `.env` file in the project root.
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## File Structure

- `app.py`: Main application file for Streamlit.
- `requirements.txt`: Contains all required Python libraries.
- `.env`: Environment variables for sensitive credentials.
- `faiss_index/`: Directory where the FAISS vector store is saved.

## Usage

1. **Upload PDFs:**
   - Use the sidebar to upload one or multiple PDF files.
   - Click "Submit & Process" to preprocess the files.

2. **Ask Questions:**
   - Use the chat interface to ask questions about the uploaded PDFs.
   - The chatbot will provide detailed answers based on the content.

## Key Functions

### PDF Text Processing
- **`get_pdf_text(pdf_docs):`** Extracts text from PDF files.

### Text Chunking
- **`get_text_chunks(text):`** Splits extracted text into manageable chunks.

### Vector Store Management
- **`get_vector_store(chunks):`** Embeds text chunks and stores them in FAISS.

### Conversational Chain
- **`get_conversational_chain():`** Configures a chain for question-answering using Gemini-Pro and LangChain.

### User Input Processing
- **`user_input(user_question):`** Handles user queries and retrieves relevant answers from the vector store.

## Example Queries

- "What is the summary of the document?"
- "Find details about [specific topic] from the uploaded PDFs."
- "Explain the section on [topic]."

## Known Issues

- Some PDF files with complex formatting may not extract text properly.
- The chatbot relies heavily on the context provided in the PDFs; answers outside the provided context will not be generated.


# Conversational RAG System

A simple conversational Retrieval Augmented Generation (RAG) system built with LangChain, ChromaDB, and Gradio. This application allows you to upload documents (PDF, DOCX, Excel) and have natural conversations with them using AI.

## Features

- Support for multiple document formats: PDF, DOCX, and Excel (XLSX/XLS)
- Vector-based document retrieval using ChromaDB
- Conversational interface with chat history
- Source attribution for answers
- User-friendly Gradio web interface
- Local embedding model (no API required for embeddings)

## Requirements

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Navigate to the project directory:
```bash
cd Part-2/Project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:7860
```

3. Using the application:
   - Enter your OpenAI API key in the designated field
   - Upload one or more documents (PDF, DOCX, or Excel)
   - Click "Process Documents" to index them
   - Start asking questions about your documents in the chat interface

## File Structure

```
Project/
├── app.py                  # Main Gradio application
├── rag_system.py          # RAG system implementation
├── document_loaders.py    # Document loading utilities
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── chroma_db/            # ChromaDB storage (created automatically)
```

## Components

### Document Loaders (`document_loaders.py`)
Handles loading and processing of different document types:
- PDF: Extracts text page by page
- DOCX: Extracts text from paragraphs
- Excel: Converts spreadsheet data to text format

### RAG System (`rag_system.py`)
Core RAG functionality:
- Document processing and chunking
- Vector storage using ChromaDB
- Conversational retrieval chain with memory
- Uses HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)
- OpenAI GPT-3.5-turbo for generating responses

### Gradio Interface (`app.py`)
Web-based user interface:
- Document upload interface
- Chat interface with history
- Clear and reset functionality
- Source attribution display

## Configuration

### Text Splitting
- Chunk size: 1000 characters
- Chunk overlap: 200 characters

### Retrieval
- Number of retrieved documents: 3
- Embedding model: sentence-transformers/all-MiniLM-L6-v2

### LLM
- Model: GPT-3.5-turbo
- Temperature: 0.7

## API Key

You need an OpenAI API key to use this application. You can:
1. Enter it in the web interface
2. Set it as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Troubleshooting

### ChromaDB Issues
If you encounter ChromaDB errors, try deleting the `chroma_db` directory and restart the application.

### File Upload Errors
Ensure your files are not corrupted and are in supported formats (PDF, DOCX, XLSX).

### Memory Issues
For large documents, the system might use significant memory. Consider:
- Processing documents in smaller batches
- Reducing chunk size in `rag_system.py`

## Limitations

- Requires internet connection for OpenAI API calls
- Works best with text-based documents
- Excel files are converted to text representation
- Performance depends on document size and complexity

## Future Enhancements

- Support for more document formats
- Option to use local LLMs
- Advanced retrieval strategies
- Export chat history
- Multi-language support

## License

This project is created for educational purposes as part of the IBM AI Engineer course.


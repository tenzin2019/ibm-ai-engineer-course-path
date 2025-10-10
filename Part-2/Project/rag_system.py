"""
RAG (Retrieval Augmented Generation) System using LangChain and ChromaDB
"""
import os
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from document_loaders import DocumentLoader


class RAGSystem:
    """Conversational RAG system with document processing and retrieval"""
    
    def __init__(self, openai_api_key: str = None, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system
        
        Args:
            openai_api_key: OpenAI API key for LLM
            persist_directory: Directory to persist ChromaDB
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize vector store
        self.vector_store = None
        self.conversation_chain = None
        self.chat_history = []
        
    def process_documents(self, file_paths: List[str]) -> str:
        """
        Process multiple documents and add them to the vector store
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Status message
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = DocumentLoader.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                return f"Error processing {file_path}: {str(e)}"
        
        if not all_documents:
            return "No documents were loaded successfully."
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(all_documents)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store.add_documents(splits)
        
        # Initialize conversation chain
        self._initialize_conversation_chain()
        
        return f"Successfully processed {len(file_paths)} file(s) with {len(splits)} chunks."
    
    def _initialize_conversation_chain(self):
        """Initialize the conversational retrieval chain"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key
        )
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create conversational retrieval chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Reset chat history
        self.chat_history = []
    
    def chat(self, question: str) -> Tuple[str, List[Document]]:
        """
        Chat with the RAG system
        
        Args:
            question: User question
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if self.conversation_chain is None:
            return "Please upload documents first before asking questions.", []
        
        try:
            result = self.conversation_chain({
                "question": question,
                "chat_history": self.chat_history
            })
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Update chat history
            self.chat_history.append((question, answer))
            
            return answer, source_docs
        
        except Exception as e:
            return f"Error processing question: {str(e)}", []
    
    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []
        if self.conversation_chain:
            self._initialize_conversation_chain()
    
    def reset(self):
        """Reset the entire system"""
        self.vector_store = None
        self.conversation_chain = None
        self.chat_history = []


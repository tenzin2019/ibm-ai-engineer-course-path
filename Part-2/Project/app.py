"""
Conversational RAG Application with Gradio Interface
"""
import os
import gradio as gr
from typing import List, Tuple
from rag_system import RAGSystem


# Initialize RAG system
rag = RAGSystem()


def process_files(files, api_key: str) -> str:
    """
    Process uploaded files and add them to the vector store
    
    Args:
        files: List of uploaded files
        api_key: OpenAI API key
        
    Returns:
        Status message
    """
    if not files:
        return "Please upload at least one file."
    
    if not api_key:
        return "Please provide your OpenAI API key."
    
    # Update API key
    rag.openai_api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Get file paths
    file_paths = [file.name for file in files]
    
    # Process documents
    status = rag.process_documents(file_paths)
    
    return status


def chat_interface(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Handle chat interaction
    
    Args:
        message: User message
        history: Chat history
        
    Returns:
        Updated history
    """
    if not message.strip():
        return history
    
    # Get response from RAG system
    answer, source_docs = rag.chat(message)
    
    # Format response with sources
    response = answer
    if source_docs:
        response += "\n\n**Sources:**\n"
        for i, doc in enumerate(source_docs[:3], 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            sheet = doc.metadata.get("sheet", "")
            
            source_info = f"{i}. {os.path.basename(source)}"
            if page:
                source_info += f" (Page {page})"
            elif sheet:
                source_info += f" (Sheet: {sheet})"
            
            response += f"\n{source_info}"
    
    return response


def clear_chat():
    """Clear chat history"""
    rag.clear_history()
    return None


def reset_system():
    """Reset the entire system"""
    rag.reset()
    return "System reset successfully. Please upload new documents.", None


# Create Gradio interface
with gr.Blocks(title="Conversational RAG System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Conversational RAG System
        Upload your documents (PDF, DOCX, Excel) and chat with them using AI.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Document Upload")
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-...",
                type="password",
                lines=1
            )
            file_input = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".doc", ".xlsx", ".xls"]
            )
            upload_btn = gr.Button("Process Documents", variant="primary")
            status_output = gr.Textbox(
                label="Status",
                lines=3,
                interactive=False
            )
            
            gr.Markdown("### Actions")
            clear_btn = gr.Button("Clear Chat History")
            reset_btn = gr.Button("Reset System", variant="stop")
        
        with gr.Column(scale=2):
            gr.Markdown("### Chat Interface")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True
            )
            msg_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about your documents...",
                lines=2
            )
            submit_btn = gr.Button("Send", variant="primary")
    
    gr.Markdown(
        """
        ### Instructions:
        1. Enter your OpenAI API key
        2. Upload one or more documents (PDF, DOCX, or Excel)
        3. Click "Process Documents" to index them
        4. Start asking questions about your documents in the chat interface
        
        **Supported File Types:** PDF, DOCX, XLSX
        """
    )
    
    # Event handlers
    upload_btn.click(
        fn=process_files,
        inputs=[file_input, api_key_input],
        outputs=status_output
    )
    
    msg_input.submit(
        fn=chat_interface,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    ).then(
        lambda: "",
        outputs=msg_input
    )
    
    submit_btn.click(
        fn=chat_interface,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    ).then(
        lambda: "",
        outputs=msg_input
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=chatbot
    )
    
    reset_btn.click(
        fn=reset_system,
        outputs=[status_output, chatbot]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


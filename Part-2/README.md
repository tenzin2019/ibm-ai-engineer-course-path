# Part 2: Advanced Natural Language Processing & Transformer Models

## 📚 Overview

Part 2 of the IBM AI Engineer course focuses on advanced Natural Language Processing (NLP) techniques, transformer architectures, and modern deep learning approaches for language understanding and generation. This comprehensive section covers everything from basic tokenization to building production-ready RAG applications with advanced fine-tuning techniques.

## 🗓️ Course Progression Timeline

| Week | Lab | Focus Area | Key Deliverables |
|------|-----|------------|------------------|
| 1 | Lab 0-1 | Foundation | Tokenization & Data Processing |
| 2 | Lab 2-3 | Neural Networks | Document Classification & Seq2Seq |
| 3 | Lab 4-5 | Transformers | Attention & Pre-training |
| 4 | Lab 6 | Fine-tuning | LoRA, QLoRA, Adapters |
| 5 | Lab 7 | Advanced Fine-tuning | RLHF, DPO, Instruction Tuning |
| 6 | Lab 8 | RAG Systems | LangChain & Retrieval |
| 7 | Lab 9 | Final Project | Production RAG Application |

## 🎯 Learning Objectives

By completing Part 2, you will be able to:

- Implement and understand tokenization techniques
- Build and train neural networks for NLP tasks
- Understand and implement transformer architectures
- Work with sequence-to-sequence models and Word2Vec
- Pre-train and fine-tune transformer models
- Use advanced techniques like LoRA, QLoRA, and adapters
- Master advanced fine-tuning including instruction tuning, RLHF, and DPO
- Build and deploy RAG (Retrieval-Augmented Generation) systems
- Integrate LangChain for complex AI applications
- Create production-ready AI applications with Gradio interfaces
- Deploy and optimize transformer models for production

## 📁 Course Structure

### Lab 0: Generative AI Introduction
**📂 Directory:** `Lab 0 - GenAI Intro/`

**🎯 Objective:** Introduction to generative AI libraries and tools

**📄 Contents:**
- `Exploring Generative AI Libraries-v2.ipynb` - Comprehensive exploration of modern AI libraries

**🔧 Skills Covered:**
- Understanding generative AI landscape
- Exploring popular AI libraries
- Setting up development environment

---

### Lab 1: Tokenization Fundamentals
**📂 Directory:** `Lab 1 - Tokenization/`

**🎯 Objective:** Master text tokenization techniques and data preprocessing

**📄 Contents:**
- `1 Implementing Tokenization-v2.ipynb` - Core tokenization implementation
- `2 Creating_an_NLP_Data_Loader_.ipynb` - Data loading and preprocessing
- `data/` - Training and validation datasets
- `Readings.png` - Supplementary reading materials

**🔧 Skills Covered:**
- Text tokenization techniques
- Data preprocessing pipelines
- Custom tokenizer implementation
- Dataset preparation for NLP tasks

---

### Lab 2: Neural Networks for NLP
**📂 Directory:** `Lab 2 - Neural Networks/`

**🎯 Objective:** Build and train neural networks for document classification and language modeling

**📄 Contents:**
- `Step 1 Classifying_Document.ipynb` - Document classification with neural networks
- `Step 2 LanguageModelling-v1.ipynb` - Language modeling implementation
- `Step 3 FeedForwardNeuralNetworks-v1.ipynb` - Feed-forward neural networks
- `*.pth` files - Pre-trained model weights (2gram, 4gram, 8gram, my_model)

**🔧 Skills Covered:**
- Document classification
- Language modeling
- Feed-forward neural networks
- Model training and evaluation
- N-gram language models

---

### Lab 3: Sequence-to-Sequence Neural Networks
**📂 Directory:** `Lab 3 - Seq2Seg NN/`

**🎯 Objective:** Implement sequence-to-sequence models and Word2Vec embeddings

**📄 Contents:**
- `RNN Encoder Decoder/` - Sequence-to-sequence model implementation
- `Word2Vec/` - Word2Vec embedding techniques
- `Evaluation Metrices for NLP Model.pdf` - Comprehensive evaluation metrics
- `Evaluation Metrrics for NLP.png` - Visual guide to evaluation metrics
- `Cheat sheet.pdf` - Quick reference guide
- `Glossary.pdf` - Key terminology and concepts

**🔧 Skills Covered:**
- RNN encoder-decoder architectures
- Word2Vec embeddings
- Sequence-to-sequence modeling
- NLP evaluation metrics
- Attention mechanisms

---

### Lab 4: Transformer Architecture
**📂 Directory:** `Lab 4 - Transformer Architecture/`

**🎯 Objective:** Deep dive into transformer architecture and attention mechanisms

**📄 Contents:**
- `Attention Mechanism and Positional Encoding.ipynb` - Core transformer concepts
- `Breaking_Down_the_Transformer_for_Classification.ipynb` - Transformer implementation

**🔧 Skills Covered:**
- Attention mechanisms
- Positional encoding
- Transformer architecture
- Multi-head attention
- Transformer for classification tasks

---

### Lab 5: Pre-training and Language Models
**📂 Directory:** `Lab 5/`

**🎯 Objective:** Pre-train BERT and implement decoder-based language models

**📄 Contents:**
- `1 Decoder Casual Language Model.ipynb` - Causal language modeling
- `2 Pre-training_BERT.ipynb` - BERT pre-training implementation
- `3 Data loading and Text processing fro BERT.ipynb` - BERT data preparation
- `4 Tranformer model Translation Traning.ipynb` - Translation model training
- `Cheat sheet.pdf` - Quick reference guide
- `Glossary.pdf` - Key terminology

**🔧 Skills Covered:**
- BERT pre-training
- Causal language modeling
- Data preprocessing for transformers
- Translation model training
- Masked language modeling

---

### Lab 6: Fine-tuning Transformers
**📂 Directory:** `Lab 6 - Fine Tuning Transformers/`

**🎯 Objective:** Master fine-tuning techniques for transformer models

**📄 Contents:**
- `1 HuggingFace Inference-v1.ipynb` - Model inference with HuggingFace
- `2 Pre-training LLMs with Hugging Face-v1.ipynb` - LLM pre-training
- `3 Fine-tuning a transformer-based NN with pytorch-v1.ipynb` - PyTorch fine-tuning
- `4 Fine-tuning Transformers with HuggingFace-v1.ipynb` - HuggingFace fine-tuning
- `5 Adapters in PyTorch-v1.ipynb` - Adapter-based fine-tuning
- `6 Enhance Model Generalization with LoRA- From AG News to IMDB-v1.ipynb` - LoRA implementation
- `7 QLoRA with Hugging Face.ipynb` - Quantized LoRA
- `8 Soft Prompts.pdf` - Soft prompt tuning
- `Cheat Sheet.pdf` - Quick reference guide
- `Glossary.pdf` - Key terminology
- `*.pt` files - Pre-trained model weights
- `results/` - Training results and checkpoints
- `trained_model/` - Fine-tuned models
- `*.txt` files - WikiText datasets

**🔧 Skills Covered:**
- HuggingFace transformers library
- Model fine-tuning techniques
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Adapter-based fine-tuning
- Soft prompt tuning
- Model evaluation and deployment

---

### Lab 7: Advanced Fine-tuning Transformers
**📂 Directory:** `Lab 7 - Advanced Fine Tuning Transformers/`

**🎯 Objective:** Master advanced fine-tuning techniques including instruction tuning, RLHF, and DPO

**📄 Contents:**
- `0 Instruction Tunning.pdf` - Instruction tuning fundamentals
- `1 Instruction fine-tuning-v1.ipynb` - Instruction-based fine-tuning implementation
- `2 Best Practices for Instruction-Tuning Large Language Models | Coursera.pdf` - Best practices guide
- `3 Reward Modeling and Response Eval.pdf` - Reward modeling concepts
- `4 RewardTrainer-v1.ipynb` - Reward model training
- `5 Log Dervative trick.pdf` - Mathematical foundations
- `6 PPOTrainer-v1.ipynb` - Proximal Policy Optimization training
- `7 Summary and Highlights | Coursera.pdf` - Course summary
- `8 DPO Fine-Tuning-v1.ipynb` - Direct Preference Optimization
- `9 Fine-Tune LLMs Locally with InstructLab.pdf` - Local fine-tuning with InstructLab
- `10 Cheat Sheet- Generative AI Advanced Fine-Tuning for LLMs.pdf` - Quick reference
- `11 Glossary- Generative AI Advance Fine-Tuning for LLMs.pdf` - Key terminology
- `CodeAlpaca-20k.json` - Instruction dataset

**🔧 Skills Covered:**
- Instruction tuning and alignment
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- Reward modeling and evaluation
- Advanced fine-tuning strategies
- Local model training with InstructLab

---

### Lab 8: RAG Fundamentals
**📂 Directory:** `Lab 8 RAG Fundamentals/`

**🎯 Objective:** Master Retrieval-Augmented Generation (RAG) systems and LangChain integration

**📄 Contents:**
- `1 RAG-v1.ipynb` - RAG fundamentals and implementation
- `2 RAG with Pytorch-v1.ipynb` - PyTorch-based RAG system
- `3 In-Context Learning and Prompt Templates for Advanced AI.ipynb` - Prompt engineering
- `4 IBM introduction to langchain-v1.ipynb` - LangChain introduction
- `5 Summarize private documents using RAG LangChain and LLMs.ipynb` - Document summarization
- `prompt-engineering-v1.ipynb` - Advanced prompt engineering
- `Cheat Sheet.pdf` - Quick reference guide
- `Reading-Glossary.pdf` - Comprehensive glossary
- `companyPolicies*.txt` - Sample documents for RAG
- `model_cache/` - Cached models (DPR encoders, GPT-2)

**🔧 Skills Covered:**
- Retrieval-Augmented Generation (RAG)
- LangChain framework integration
- Document processing and chunking
- Vector databases and embeddings
- Prompt engineering techniques
- In-context learning
- Document summarization with RAG
- Dense Passage Retrieval (DPR)

---

### Lab 9: Final Project - Generative AI Applications
**📂 Directory:** `Lab 9 - Final Project/`

**🎯 Objective:** Build a complete RAG-based question-answering system using LangChain and watsonx

**📄 Contents:**
- `1 LangChain document loader.ipynb` - Document loading with LangChain
- `2 LangChain text-splitter.ipynb` - Text chunking strategies
- `3 Embed Documents Using watsonx's Embedding Model.pdf` - watsonx embedding guide
- `4 Embed documents with watsonx s embedding.ipynb` - Document embedding implementation
- `5 LangChain vector store.ipynb` - Vector store setup
- `6 LangChain retriever.ipynb` - Retrieval system implementation
- `7 Reading - Compare Fine-Tuning Using InstructLab with RAG.pdf` - Fine-tuning vs RAG comparison
- `8 Gradio Setup and Tutorial.pdf` - Gradio interface guide
- `8.1Gradio_Practice.py` - Gradio practice exercises
- `9 Project Overview.pdf` - Complete project overview
- `9.1 qabot.py` - Final Q&A bot implementation
- `Cheat Sheet- Project- Generative AI Applications with RAG and LangChain.pdf` - Project reference
- `Glossary.pdf` - Key terminology

**🔧 Skills Covered:**
- End-to-end RAG system development
- LangChain framework mastery
- watsonx integration
- Document processing pipeline
- Vector store management
- Retrieval system optimization
- Gradio web interface development
- Production-ready AI application deployment

## 🛠️ Technical Requirements

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Transformers library
- Datasets library
- TRL (Transformer Reinforcement Learning)
- Accelerate library

### Key Libraries
```bash
pip install torch torchvision torchtext
pip install transformers datasets
pip install trl accelerate
pip install torchmetrics
pip install sentencepiece tokenizers
pip install langchain langchain-community
pip install chromadb faiss-cpu
pip install gradio
pip install watsonx-ai
```

### Hardware Requirements
- **Minimum:** 8GB RAM, CPU training
- **Recommended:** 16GB+ RAM, GPU with 8GB+ VRAM
- **Optimal:** Multi-GPU setup for large model training

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv ibm-ai-venv
source ibm-ai-venv/bin/activate  # On Windows: ibm-ai-venv\Scripts\activate

# Install core dependencies
pip install torch torchvision torchtext
pip install transformers datasets
pip install trl accelerate
pip install langchain langchain-community
pip install gradio
pip install watsonx-ai
```

### 2. Quick Start Guide
```bash
# Clone and navigate to Part 2
cd Part-2

# Start with Lab 0 for an overview
jupyter notebook "Lab 0 - GenAI Intro/Exploring Generative AI Libraries-v2.ipynb"

# Follow the progression timeline above
# Each lab builds upon previous concepts
```

### 3. Data Preparation
- Download required datasets (instructions in each lab)
- Ensure sufficient disk space for model weights and datasets (50GB+ recommended)
- Set up proper data paths in notebooks
- Configure API keys for watsonx (Lab 9)

### 4. Model Training
- Start with Lab 1 for foundational concepts
- Progress through labs sequentially
- Use provided pre-trained models for faster experimentation
- Monitor GPU memory usage during training

## 📊 Key Concepts Covered

### 1. Tokenization
- Byte-pair encoding (BPE)
- WordPiece tokenization
- SentencePiece
- Custom tokenizer implementation

### 2. Neural Networks
- Feed-forward networks
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

### 3. Transformer Architecture
- Multi-head attention
- Positional encoding
- Self-attention mechanisms
- Encoder-decoder architecture

### 4. Pre-training Techniques
- Masked Language Modeling (MLM)
- Causal Language Modeling (CLM)
- Next Sentence Prediction (NSP)
- Span corruption

### 5. Fine-tuning Methods
- Full fine-tuning
- Parameter-efficient fine-tuning (PEFT)
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Adapter-based fine-tuning
- Soft prompt tuning
- Instruction tuning and alignment
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)

### 6. RAG and Advanced Applications
- Retrieval-Augmented Generation (RAG)
- LangChain framework integration
- Document processing and chunking
- Vector databases and embeddings
- Dense Passage Retrieval (DPR)
- In-context learning
- Prompt engineering techniques
- Production deployment with Gradio

## 🎓 Assessment & Evaluation

### Practical Projects
- Document classification system
- Language model implementation
- Translation model training
- Fine-tuned chatbot development
- Advanced instruction-tuned models
- RAG-based question-answering system
- Production-ready AI application with Gradio interface

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Perplexity for language models
- BLEU score for translation
- ROUGE score for summarization

## 🔧 Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or use gradient accumulation
2. **CUDA Out of Memory**: Enable gradient checkpointing or use smaller models
3. **Tokenization Errors**: Ensure proper padding token configuration
4. **Training Instability**: Adjust learning rate and warmup steps

### Performance Optimization
- Use mixed precision training (fp16/bf16)
- Implement gradient accumulation
- Enable gradient checkpointing
- Use efficient data loading

## 📚 Additional Resources

### Documentation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TRL Documentation](https://huggingface.co/docs/trl/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Gradio Documentation](https://gradio.app/docs/)
- [IBM watsonx Documentation](https://dataplatform.cloud.ibm.com/docs/)

### Research Papers
- "Attention Is All You Need" - Transformer architecture
- "BERT: Pre-training of Deep Bidirectional Transformers" - BERT paper
- "LoRA: Low-Rank Adaptation of Large Language Models" - LoRA technique
- "Training language models to follow instructions with human feedback" - RLHF paper
- "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" - DPO paper
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" - RAG paper

### Community Resources
- HuggingFace Hub for pre-trained models
- Papers With Code for implementations
- GitHub repositories for advanced techniques
- LangChain Hub for prompt templates
- Gradio Spaces for sharing applications
- IBM watsonx Community for enterprise AI solutions

## 🎯 Next Steps

After completing Part 2, you'll be ready to:
- Deploy transformer models in production
- Implement advanced NLP applications with RAG
- Build enterprise-grade AI solutions with watsonx
- Create interactive AI applications with Gradio
- Contribute to open-source AI projects
- Pursue specialized AI engineering roles
- Lead AI product development teams

## 📞 Support

For technical support or questions:
- Check the troubleshooting section in each lab
- Review the cheat sheets and glossaries provided
- Consult the official documentation links
- Join the IBM AI Engineer community forums

---

## 🏆 Course Completion

Upon successful completion of all labs, you will have:

- ✅ Mastered modern NLP and transformer architectures
- ✅ Built production-ready AI applications
- ✅ Implemented advanced fine-tuning techniques
- ✅ Created RAG systems with LangChain
- ✅ Deployed interactive AI interfaces with Gradio
- ✅ Gained hands-on experience with enterprise AI tools

## 🎓 Certification

This course is part of the IBM AI Engineer Professional Certificate program. Complete all labs and assessments to earn your certification.

---

**Happy Learning! 🚀**

*This comprehensive course provides a complete foundation in modern NLP and transformer technologies, preparing you for real-world AI engineering challenges and career advancement in the field of artificial intelligence.*

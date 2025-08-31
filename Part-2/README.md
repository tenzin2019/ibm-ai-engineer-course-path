# Part 2: Advanced Natural Language Processing & Transformer Models

## 📚 Overview

Part 2 of the IBM AI Engineer course focuses on advanced Natural Language Processing (NLP) techniques, transformer architectures, and modern deep learning approaches for language understanding and generation. This section covers everything from basic tokenization to fine-tuning large language models.

## 🎯 Learning Objectives

By completing Part 2, you will be able to:

- Implement and understand tokenization techniques
- Build and train neural networks for NLP tasks
- Understand and implement transformer architectures
- Work with sequence-to-sequence models and Word2Vec
- Pre-train and fine-tune transformer models
- Use advanced techniques like LoRA, QLoRA, and adapters
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

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
- Download required datasets (instructions in each lab)
- Ensure sufficient disk space for model weights and datasets
- Set up proper data paths in notebooks

### 3. Model Training
- Start with Lab 1 for foundational concepts
- Progress through labs sequentially
- Use provided pre-trained models for faster experimentation

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

## 🎓 Assessment & Evaluation

### Practical Projects
- Document classification system
- Language model implementation
- Translation model training
- Fine-tuned chatbot development

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

### Research Papers
- "Attention Is All You Need" - Transformer architecture
- "BERT: Pre-training of Deep Bidirectional Transformers" - BERT paper
- "LoRA: Low-Rank Adaptation of Large Language Models" - LoRA technique

### Community Resources
- HuggingFace Hub for pre-trained models
- Papers With Code for implementations
- GitHub repositories for advanced techniques

## 🎯 Next Steps

After completing Part 2, you'll be ready to:
- Deploy transformer models in production
- Implement advanced NLP applications
- Contribute to open-source AI projects
- Pursue specialized AI engineering roles

## 📞 Support

For technical support or questions:
- Check the troubleshooting section in each lab
- Review the cheat sheets and glossaries provided
- Consult the official documentation links
- Join the IBM AI Engineer community forums

---

**Happy Learning! 🚀**

*This course provides a comprehensive foundation in modern NLP and transformer technologies, preparing you for real-world AI engineering challenges.*

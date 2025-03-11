# Resume Analyzer AI - AI Model Fine-Tuning

Resume Analyzer AI is a demo project showcasing how to train and fine-tune an AI model for resume analysis. This project fine-tunes the Llama 3.1 8B model on a structured resume dataset to demonstrate:

- How to fine-tune Llama 3.1 8B using LoRA
- How to prepare datasets for AI training
- How to optimize training for limited VRAM (RTX 4070 Ti)

## Project Focus

- ✅ AI Model Training & Fine-Tuning
- ✅ Dataset Preprocessing for Resume Analysis
- ✅ Optimizing LLM Training with Low VRAM
- ✅ Hands-on Demonstration of Fine-Tuning Techniques

## Tech Stack

- Model: Llama 3.1 8B (Fine-tuned with LoRA)
- Frameworks: PyTorch, Hugging Face Transformers
- Dataset: Structured Resume Dataset (1800+ entries)
- Hardware: Optimized for i9 13900K, 32GB RAM and RTX 4070 Ti

## Requirements
- Python 3.10+
- Hugging Face CLI installed (`pip install huggingface_hub`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resume-analyzer-ai.git
   cd resume-analyzer-ai
   ```

2. Run setup:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. Login to Hugging Face
   ```bash
   huggingface-cli login
   ```

## Training
To fine-tune the model:
```bash
python train.py
```
Note: Training will take time, and checkpoints will be saved every 500 steps.

## Resume Training (if needed):
```bash
python train.py --resume_from_checkpoint models/checkpoints/latest
```

## Testing
To analyze a sample resume:
```bash
python inference.py
```

## Contributing
This project serves as a learning resource for fine-tuning AI models. Feel free to submit pull requests or open issues to improve the training process.

## Show Your Support!
If you like this project, give it a star ⭐ on GitHub!
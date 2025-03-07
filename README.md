# Resume Analyzer AI

This project fine-tunes Llama 3.1 8B to analyze resumes, extract key information, and suggest job roles.

## Features
✅ Summarizes skills, experience, and education  
✅ Scores resumes based on job role matching  
✅ Suggests best-fitting job roles  

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
python inference.py --resume "sample_resume.txt"
```
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import constants

print("CUDA available:", torch.cuda.is_available())

model_name = constants.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

dataset = load_dataset("json", data_files=constants.process_file_path)

def tokenize_function(example):
    return tokenizer(example["input"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=constants.model_checkpoint_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=500,
    evaluation_strategy="epoch",
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

if __name__ == "__main__":
    print(f"Started training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    model.save_pretrained(constants.trained_model_path)
    tokenizer.save_pretrained(constants.trained_model_path)
    print(f"Completed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
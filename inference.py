from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import sys

model_path = "models/fine_tuned_llama_resume"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")

def analyze_resume(resume_text):
    inputs = tokenizer(f"Analyze this resume:\n\n{resume_text}", return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    resume_text = sys.argv[1] if len(sys.argv) > 1 else "Sample resume text here."
    print(analyze_resume(resume_text))

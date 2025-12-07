import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge_lora():
    # Paths
    base_model_path = "models/Qwen3-1.7B"
    lora_adapter_path = "models/lora-qwen3-1.7B"
    output_path = "models/merged-qwen3-1.7B"

    print(f"Loading base model from {base_model_path}...")
    # Load base model in float16 to avoid memory issues and for better compatibility
    # device_map="auto" caused issues with offloading and Peft loading (KeyError)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from {lora_adapter_path}...")
    
    # Resize embeddings if necessary to match LoRA checkpoint
    print("Resizing token embeddings to match LoRA checkpoint (151669)...")
    base_model.resize_token_embeddings(151669)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("Merging weights...")
    # Merge and unload
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    # Save merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Done!")

if __name__ == "__main__":
    # Ensure we are in the correct directory (NLP-Final)
    if os.path.basename(os.getcwd()) != "NLP-Final":
        if os.path.exists("NLP-Final"):
            os.chdir("NLP-Final")
            print("Changed directory to NLP-Final")
        else:
            print("Warning: NLP-Final directory not found, assuming current directory structure matches script expectations.")

    merge_lora()


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AutoModel
from datasets import load_dataset
import logging
import os
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoTDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoded = self.tokenizer(item['input_text'], item['output_text'], 
                                 truncation=True, max_length=self.max_length, 
                                 padding='max_length', return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }

class ModelEnhancementFramework:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = None
        self.tokenizer = None
        self.cot_prompt = "Let's approach this step-by-step:\n1)"
        self.device = device
        logging.info(f"Using device: {self.device}")

    def load_model(self, model_name):
        logging.info(f"Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def apply_chain_of_thought(self):
        logging.info("Applying chain of thought...")
        try:
            cot_dataset = load_dataset("bigbench", "strategyqa", split="train")
            
            def format_cot(example):
                return {
                    "input_text": f"{self.cot_prompt} {example['input']}",
                    "output_text": example['target']
                }
            
            cot_dataset = cot_dataset.map(format_cot)
            
            logging.info("Fine-tuning model on CoT dataset...")
            
            train_dataset = CoTDataset(self.tokenizer, cot_dataset, max_length=512)
            
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                save_steps=10_000,
                save_total_limit=2,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()
            logging.info("Fine-tuning completed.")
            
        except Exception as e:
            logging.error(f"Error in applying chain of thought: {e}")
            raise

    def apply_reflection_mechanism(self):
        logging.info("Applying reflection mechanism...")
        # In this implementation, we're using the same model for reflection
        # A more advanced implementation could use a separate model or mechanism

    @torch.no_grad()
    def reflect_on_output(self, output):
        reflection_prompt = f"Evaluate the following response. If it's high quality, return 'GOOD'. If it needs improvement, return 'IMPROVE': {output}"
        inputs = self.tokenizer(reflection_prompt, return_tensors="pt").to(self.device)
        try:
            with autocast():
                reflection_output = self.model.generate(**inputs, max_length=50)
            reflection = self.tokenizer.decode(reflection_output[0], skip_special_tokens=True)
            
            if "GOOD" in reflection:
                return output
            else:
                improvement_prompt = f"Improve this response: {output}"
                inputs = self.tokenizer(improvement_prompt, return_tensors="pt").to(self.device)
                with autocast():
                    improved_output = self.model.generate(**inputs, max_length=200)
                return self.tokenizer.decode(improved_output[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error in reflection process: {e}")
            return output  # Return original output if reflection fails

    @torch.no_grad()
    def generate_text(self, prompt, max_length=200):
        full_prompt = f"{self.cot_prompt} {prompt}"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        try:
            with autocast():
                outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=3, 
                                              do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
            
            candidates = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            best_candidate = max(candidates, key=lambda x: self.evaluate_response(x))
            
            return self.reflect_on_output(best_candidate)
        except Exception as e:
            logging.error(f"Error in text generation: {e}")
            return "I apologize, but I encountered an error while generating the response."

    def evaluate_response(self, response):
        # This is a simple evaluation heuristic. You might want to implement a more sophisticated one.
        return len(response.split()) + len(set(response.split()))  # Favor longer and more diverse responses

    def enhance_model(self, model_name):
        self.load_model(model_name)
        self.apply_chain_of_thought()
        self.apply_reflection_mechanism()
        return self

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logging.info(f"Model saved to {path}")

    def load_enhanced_model(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logging.info(f"Enhanced model loaded from {path}")

def get_user_model_choice():
    print("How would you like to select a model?")
    print("1. Choose from a list of popular models")
    print("2. Enter a custom HuggingFace model name")
    print("3. Load a previously enhanced model")
    
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == '1':
        return choose_from_list(), None
    elif choice == '2':
        return enter_custom_model(), None
    elif choice == '3':
        return None, enter_enhanced_model_path()
    else:
        print("Invalid choice. Defaulting to custom model entry.")
        return enter_custom_model(), None

def choose_from_list():
    models = {
        '1': 'microsoft/Phi-3.5-mini-instruct',
        '2': 'gpt2',
        '3': 'facebook/opt-350m',
        '4': 'google/flan-t5-base',
        '5': 'EleutherAI/gpt-neo-125M'
    }
    
    print("\nAvailable models:")
    for key, model in models.items():
        print(f"{key}. {model}")
    
    while True:
        choice = input("Enter the number of your chosen model: ")
        if choice in models:
            return models[choice]
        else:
            print("Invalid choice. Please try again.")

def enter_custom_model():
    print("\nEnter the HuggingFace model name you want to use.")
    print("For example: 'microsoft/Phi-3.5-mini-instruct'")
    return input("Model name: ")

def enter_enhanced_model_path():
    print("\nEnter the path to your previously enhanced model.")
    return input("Model path: ")

# Example usage
if __name__ == "__main__":
    model_name, enhanced_model_path = get_user_model_choice()
    framework = ModelEnhancementFramework()
    
    try:
        if enhanced_model_path:
            framework.load_enhanced_model(enhanced_model_path)
        else:
            enhanced_model = framework.enhance_model(model_name)
            framework.save_model("./enhanced_model")
        
        while True:
            prompt = input("Enter a prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
            response = framework.generate_text(prompt)
            print(f"Enhanced model response:\n{response}\n")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print("Please make sure you have the necessary dependencies installed and the model name/path is correct.")
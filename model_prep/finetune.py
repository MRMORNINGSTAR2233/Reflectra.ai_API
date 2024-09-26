import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers.trainer_utils import get_last_checkpoint
from torch.cuda.amp import autocast
from nomic import NomicEmbedding
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Dataset class for fine-tuning
class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx]['text']
        encoded = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()  # Labels same as input for LM fine-tuning
        }

def prepare_custom_dataset(dataset_path, tokenizer, max_length=512):
    """Prepare the custom dataset using LangChain and FAISS with Nomic embeddings."""
    logging.info(f"Loading custom dataset from {dataset_path}...")
    # Load documents
    loader = TextLoader(file_path=dataset_path)
    documents = loader.load()

    # Generate embeddings using Nomic via Ollama
    logging.info("Generating embeddings with Nomic...")
    texts = [doc.page_content for doc in documents]
    nomic_embedding = NomicEmbedding()
    embeddings = nomic_embedding.embed(texts)

    # Storing embeddings in FAISS for efficient retrieval
    logging.info("Creating FAISS vector store...")
    vector_store = FAISS.from_embeddings(embeddings, documents)

    # Retrieve data for fine-tuning
    logging.info("Preparing data for fine-tuning...")
    data = [{"text": doc.page_content} for doc in documents]
    
    return CustomTextDataset(tokenizer, data, max_length)

def fine_tune_model(model_path, dataset_path, output_dir="./fine_tuned_model"):
    """Fine-tune the model with a custom dataset, focusing on speed optimization."""
    try:
        # Load the enhanced model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Prepare the custom dataset
        dataset = prepare_custom_dataset(dataset_path, tokenizer)

        # Set up training arguments for fast fine-tuning
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,  # Reduce epochs to speed up
            per_device_train_batch_size=4,  # Adjust batch size based on GPU memory
            gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
            fp16=True,  # Mixed precision training
            save_steps=1000,  # Save more frequently
            save_total_limit=1,  # Keep only the latest checkpoint to save space
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,  # More frequent logging
            evaluation_strategy="steps",  # Evaluate at regular intervals
            eval_steps=500,  # Perform evaluation every 500 steps
            dataloader_num_workers=4,  # Increase data loading speed
            optim="adamw_torch"  # Use optimized AdamW for PyTorch
        )

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            # Only fine-tuning final layers for faster training
            freeze_encoder=True  # This argument will need to be supported by the model architecture
        )

        # Check for existing checkpoint to continue training
        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        if last_checkpoint:
            logging.info(f"Resuming training from {last_checkpoint}")
        else:
            logging.info("Starting new training run...")

        # Start fine-tuning
        logging.info("Starting fine-tuning process...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
        logging.info(f"Model fine-tuned successfully. Saved to {output_dir}")

    except Exception as e:
        logging.error(f"Error in fine-tuning process: {e}")

if __name__ == "__main__":
    # Define paths
    model_path = "./enhanced_model"  # Path to the enhanced model
    dataset_path = "path/to/your/dataset.txt"  # Replace with the path to your dataset

    # Fine-tune the model with the custom dataset
    fine_tune_model(model_path, dataset_path)

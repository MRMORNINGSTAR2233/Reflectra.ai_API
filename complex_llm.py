import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm.auto import tqdm
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class AdvancedReasoningFramework:
    def __init__(self, model_name, local_model_path=None):
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.metric = load_metric("squad")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        if self.local_model_path and os.path.exists(self.local_model_path):
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = self.model.to(self.device)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def load_dataset(self, subset_size=None):
        self.dataset = load_dataset("squad")
        if subset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(min(subset_size, len(self.dataset["train"]))))
            self.dataset["validation"] = self.dataset["validation"].select(range(min(subset_size, len(self.dataset["validation"]))))

    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            context_start = 0
            while sequence_ids[context_start] != 1 and context_start < len(sequence_ids):
                context_start += 1
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1 and context_end > 0:
                context_end -= 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (context_start >= len(offset) or
                context_end < 0 or
                offset[context_start][0] > end_char or
                offset[context_end][1] < start_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        start_logits, end_logits = predictions
        start_predictions = np.argmax(start_logits, axis=1)
        end_predictions = np.argmax(end_logits, axis=1)
        
        references = [{"id": str(i), "answers": self.dataset["validation"][i]["answers"]} 
                      for i in range(len(self.dataset["validation"]))]
        
        predictions = []
        for i, (start, end) in enumerate(zip(start_predictions, end_predictions)):
            context = self.dataset["validation"][i]["context"]
            offset_mapping = self.tokenizer(context, return_offsets_mapping=True)["offset_mapping"]
            
            # Adjust for [CLS] token
            start += 1
            end += 1
            
            # Ensure start and end are within bounds
            start = max(0, min(start, len(offset_mapping) - 1))
            end = max(0, min(end, len(offset_mapping) - 1))
            
            pred_start = offset_mapping[start][0]
            pred_end = offset_mapping[end][1]
            prediction_text = context[pred_start:pred_end]
            
            predictions.append({
                "id": str(i),
                "prediction_text": prediction_text
            })
        
        # SQuAD metrics
        squad_results = self.metric.compute(predictions=predictions, references=references)
        
        # Custom metrics
        exact_match = squad_results['exact_match']
        f1 = squad_results['f1']
        
        # Calculate loss (fixed)
        loss = np.mean(np.sum((start_logits - labels[0][:, None])**2 + (end_logits - labels[1][:, None])**2, axis=1))
        
        # Calculate precision, recall, and F1 score for start and end positions
        start_precision, start_recall, start_f1, _ = precision_recall_fscore_support(labels[0], start_predictions, average='weighted')
        end_precision, end_recall, end_f1, _ = precision_recall_fscore_support(labels[1], end_predictions, average='weighted')
        
        # Calculate accuracy for start and end positions
        start_accuracy = accuracy_score(labels[0], start_predictions)
        end_accuracy = accuracy_score(labels[1], end_predictions)
        
        return {
            'exact_match': exact_match,
            'f1': f1,
            'loss': loss,
            'start_precision': start_precision,
            'start_recall': start_recall,
            'start_f1': start_f1,
            'start_accuracy': start_accuracy,
            'end_precision': end_precision,
            'end_recall': end_recall,
            'end_f1': end_f1,
            'end_accuracy': end_accuracy
        }


    def train_model(self, num_epochs=1, batch_size=12, learning_rate=5e-5):
        tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True, remove_columns=self.dataset["train"].column_names)

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        print("Starting model training...")
        train_result = trainer.train()
        print("Training completed.")
        
        print("\nTraining metrics:")
        print(train_result.metrics)

        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        print("\nEvaluation metrics:")
        print(eval_results)

        if self.local_model_path:
            print(f"\nSaving model to {self.local_model_path}")
            trainer.save_model(self.local_model_path)
        
        return train_result, eval_results

    def process_input(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
        return answer

    def chain_of_thought(self, question, context):
        thoughts = []

        # Step 1: Analyze the question
        thoughts.append(f"1. Analyzing the question: '{question}'")
        question_tokens = self.tokenizer.tokenize(question)
        thoughts.append(f"   Key question words: {', '.join(question_tokens[:5])}")

        # Step 2: Scan the context
        thoughts.append("2. Scanning the context for relevant information")
        context_sentences = context.split('.')
        relevant_sentences = [sent for sent in context_sentences if any(word in sent.lower() for word in question_tokens)]
        thoughts.append(f"   Found {len(relevant_sentences)} potentially relevant sentences")

        # Step 3: Generate potential answers
        thoughts.append("3. Generating potential answers")
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to the same device as the model
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
        thoughts.append(f"   Most likely answer: '{answer}'")

        # Step 4: Verify the answer
        thoughts.append("4. Verifying the answer")
        answer_in_context = answer.lower() in context.lower()
        thoughts.append(f"   Answer found in context: {'Yes' if answer_in_context else 'No'}")

        # Step 5: Confidence assessment
        thoughts.append("5. Assessing confidence in the answer")
        start_logits_max = torch.max(outputs.start_logits).item()
        end_logits_max = torch.max(outputs.end_logits).item()
        confidence = (start_logits_max + end_logits_max) / 2
        thoughts.append(f"   Confidence score: {confidence:.2f}")

        return answer, thoughts

    def process_input(self, question, context):
        answer, thoughts = self.chain_of_thought(question, context)
        return answer, thoughts

# Usage
model_name = "distilbert-base-uncased-distilled-squad"
local_model_path = ""  # Specify where you want to save the model

framework = AdvancedReasoningFramework(model_name, local_model_path)

# Load a subset of the dataset (e.g., 1000 examples)
framework.load_dataset(subset_size=1000)

# Train and evaluate the model
train_result, eval_results = framework.train_model(num_epochs=1)

# Process a custom question
custom_question = "Who wrote the Declaration of Independence?"
custom_context = "The Declaration of Independence is the document in which the Thirteen Colonies of Great Britain in North America declared themselves independent states and no longer under British rule. It was drafted by Thomas Jefferson, amended by the Continental Congress, and signed on July 4, 1776."
print(f"\nProcessing custom question: {custom_question}")
custom_result, thoughts = framework.process_input(custom_question, custom_context)
print("Chain of Thought:")
for thought in thoughts:
    print(thought)
print(f"\nFinal Answer: {custom_result}")
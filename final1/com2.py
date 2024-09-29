import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from evaluate import load as load_metric
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
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def load_dataset(self, subset_size=None):
        self.dataset = load_dataset("greengerong/leetcode", split='train')
        
        if subset_size:
            self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))
        
        self.dataset = self.dataset.train_test_split(test_size=0.1)

    def preprocess_function(self, examples):
        questions = [example['title'].strip() for example in examples['question_content']]
        contexts = [example['content'] for example in examples['question_content']]
        
        inputs = self.tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            context = contexts[sample_idx]
            
            # For this dataset, we'll use the entire context as the answer
            # This is a simplification and should be adjusted based on your specific needs
            start_char = 0
            end_char = len(context)

            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
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

    # ... [rest of the methods remain the same] ...

    def process_input(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
        return answer

    def chain_of_thought(self, question, context):
        thoughts = []

        thoughts.append(f"1. Analyzing the question: '{question}'")
        question_tokens = self.tokenizer.tokenize(question)
        thoughts.append(f"   Key question words: {', '.join(question_tokens[:5])}")

        thoughts.append("2. Scanning the context for relevant information")
        context_sentences = context.split('.')
        relevant_sentences = [sent for sent in context_sentences if any(word.lower() in sent.lower() for word in question_tokens)]
        thoughts.append(f"   Found {len(relevant_sentences)} potentially relevant sentences")

        thoughts.append("3. Generating potential answers")
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
        thoughts.append(f"   Most likely answer: '{answer}'")

        thoughts.append("4. Verifying the answer")
        answer_in_context = answer.lower() in context.lower()
        thoughts.append(f"   Answer found in context: {'Yes' if answer_in_context else 'No'}")

        thoughts.append("5. Assessing confidence in the answer")
        start_logits_max = torch.max(outputs.start_logits).item()
        end_logits_max = torch.max(outputs.end_logits).item()
        confidence = (start_logits_max + end_logits_max) / 2
        thoughts.append(f"   Confidence score: {confidence:.2f}")

        return answer, thoughts

    def process_input_with_cot(self, question, context):
        answer, thoughts = self.chain_of_thought(question, context)
        return answer, thoughts

# Usage
if __name__ == "__main__":
    model_name = "Luciferio/MiniLLM-finetuned"
    local_model_path = "path/to/save/model"  # Specify where you want to save the model
    
    framework = AdvancedReasoningFramework(model_name, local_model_path)
    
    # Load a subset of the dataset (e.g., 1000 examples)
    framework.load_dataset(subset_size=1000)
    
    # Train and evaluate the model
    train_result, eval_results = framework.train_model(num_epochs=1)
    
    # Process a custom question
    custom_question = "convert-sorted-list-to-binary-search-tree"
    custom_context = ("Given the `head` of a singly linked list where elements are sorted in **ascending order**, convert _it to a_ **_height-balanced_** _binary search tree_. **Example 1:** **Input:** head = \[-10,-3,0,5,9\] **Output:** \[0,-3,9,-10,null,5\] **Explanation:** One possible answer is \[0,-3,9,-10,null,5\], which represents the shown height balanced BST. **Example 2:** **Input:** head = \[\] **Output:** \[\] **Constraints:** * The number of nodes in `head` is in the range `[0, 2 * 104]`. * `-105 <= Node.val <= 105`..")
    custom_result, thoughts = framework.process_input_with_cot(custom_question, custom_context)
    
    print("Chain of Thought:")
    for thought in thoughts:
        print(thought)
    print(f"\nFinal Answer: {custom_result}")
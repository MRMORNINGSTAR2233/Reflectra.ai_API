import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import os
import random

class AdvancedReasoningFramework:
    def __init__(self, model_name, local_model_path=None):
        if local_model_path and os.path.exists(local_model_path):
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return load_dataset("squad", split="train[:5000]")  # Load a subset for faster processing

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def chain_of_thought(self, question, context):
        prompt = f"""Context: {context}

Question: {question}

Let's approach this step-by-step:
1) First, let's understand what the question is asking.
2) Next, let's identify relevant information from the context.
3) Now, let's consider how this information relates to the question.
4) Let's reason through the problem using this information.
5) Finally, we can formulate our answer based on the context and reasoning.

Detailed step-by-step reasoning:
"""
        cot_output = self.generate(prompt, max_new_tokens=200)
        return cot_output

    def reflection(self, question, context, initial_answer):
        prompt = f"""Context: {context}

Question: {question}

Initial answer: {initial_answer}

Let's reflect on this answer and improve it:
1) Is the answer complete and directly addresses the question?
2) Does it accurately use information from the context?
3) Are there any logical flaws or missing information?
4) Can we make the explanation clearer or more concise?
5) Is there any additional relevant information from the context we can include?

Improved answer:
"""
        reflection_output = self.generate(prompt, max_new_tokens=150)
        return reflection_output

    def generate(self, prompt, max_new_tokens=100):
        output = self.generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)[0]['generated_text']
        return output.replace(prompt, "").strip()

    def process_input(self, question, context):
        initial_output = self.chain_of_thought(question, context)
        refined_output = self.reflection(question, context, initial_output)
        return refined_output

    def process_dataset_sample(self, sample_index=None):
        if sample_index is None:
            sample_index = random.randint(0, len(self.dataset) - 1)
        
        sample = self.dataset[sample_index]
        question = sample['question']
        context = sample['context']
        return self.process_input(question, context)

    def get_random_questions(self, n=5):
        samples = random.sample(range(len(self.dataset)), n)
        return [(self.dataset[i]['question'], self.dataset[i]['context']) for i in samples]

# Usage
model_name = "gpt2-large"  # Using a larger model for better performance
local_model_path = "path/to/save/model"  # Specify where you want to save the model

framework = AdvancedReasoningFramework(model_name, local_model_path=local_model_path)

# Save the model locally
framework.save_model(local_model_path)

# Get some random questions from the dataset
random_questions = framework.get_random_questions()
print("Random questions from the SQuAD dataset:")
for i, (question, context) in enumerate(random_questions, 1):
    print(f"{i}. Question: {question}")
    print(f"   Context: {context[:100]}...")  # Print first 100 characters of context

# Process a sample from the dataset
print("\nProcessing a random sample from the dataset:")
dataset_result = framework.process_dataset_sample()
print(dataset_result)

# Process a custom question (you would need to provide a relevant context for a custom question)
custom_question = "Who wrote the Declaration of Independence?"
custom_context = "The Declaration of Independence is the document in which the Thirteen Colonies of Great Britain in North America declared themselves independent states and no longer under British rule. It was drafted by Thomas Jefferson, amended by the Continental Congress, and signed on July 4, 1776."
print(f"\nProcessing custom question: {custom_question}")
custom_result = framework.process_input(custom_question, custom_context)
print(custom_result)
import os
import json
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

hf_token = "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"

class CloneDetector:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B", embedding_model: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = IndexFlatL2(384)
        self.documents: List[Dict[str, str]] = []
        
        print(f"Loading language model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", token=hf_token)
        
    def read_directory(self, directory: str):
        for filename in os.listdir(directory):
            if filename.endswith('.py'):
                with open(os.path.join(directory, filename), 'r') as file:
                    content = file.read()
                    self.documents.append({
                        'id': filename,
                        'content': content
                    })
                    embedding = self.embedding_model.encode([content])[0]
                    self.index.add(np.array([embedding]).astype('float32'))
        print(f"Processed {len(self.documents)} documents")

    def detect_clones(self, query: str, k: int = 5):
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)

        retrieved_docs = [self.documents[i] for i in indices[0]]

        context = "\n".join([f"Document {doc['id']}:\n{doc['content']}\n" for doc in retrieved_docs])

        prompt = f"""Analyze the following code snippets for potential clones or similarities:

        {context}

        Provide your analysis in a JSON format with the following structure:
        {{
            "clone_detected": boolean,
            "similarity_level": "high" | "medium" | "low" | "none",
            "similarities": [
                {{
                    "type": "exact" | "near-exact" | "logical",
                    "description": "string",
                    "affected_files": ["file1", "file2", ...]
                }},
                ...
            ],
            "explanation": "string"
        }}

        Your response must be a valid JSON object. Do not include any text before or after the JSON object.
        If no clones are detected, set "clone_detected" to false, "similarity_level" to "none", and provide an empty list for "similarities".
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        max_length=2048,  # Increased max_length
                        min_length=100,   # Adjusted min_length
                        do_sample=True,
                        temperature=0.7,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        top_k=50,
                        top_p=0.95,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(output[0], skip_special_tokens=True)

                # Attempt to parse the entire response as JSON
                try:
                    json_response = json.loads(response)
                    return json_response
                except json.JSONDecodeError:
                    # If parsing the entire response fails, try to extract JSON
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response[json_start:json_end]
                        json_response = json.loads(json_str)
                        return json_response
                    else:
                        print(f"Attempt {attempt + 1}: No valid JSON object found in the response")
                        continue
            except Exception as e:
                print(f"Attempt {attempt + 1}: Unexpected error: {e}")

        # If all attempts fail, return a default response
        return {
            "error": "Failed to generate valid JSON response after multiple attempts",
            "clone_detected": False,
            "similarity_level": "none",
            "similarities": [],
            "explanation": "Unable to perform analysis due to generation error.",
            "raw_response": response
        }#     def detect_clones(self, query: str, k: int = 5):
#         query_embedding = self.embedding_model.encode([query])[0]
#         distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
#         retrieved_docs = [self.documents[i] for i in indices[0]]
        
#         context = "\n".join([f"Document {doc['id']}:\n{doc['content']}\n" for doc in retrieved_docs])
        
#         prompt = f"""Analyze the following code snippets for potential clones or similarities:

#         {context}

#         Provide your analysis in a JSON format with the following structure:
#         {{
#             "clone_detected": boolean,
#             "similarity_level": "high" | "medium" | "low" | "none",
#             "similarities": [
#                 {{
#                     "type": "exact" | "near-exact" | "logical",
#                     "description": "string",
#                     "affected_files": ["file1", "file2", ...]
#                 }},
#                 ...
#             ],
#             "explanation": "string"
#         }}

#         Ensure that your response is a valid JSON object and nothing else. only. Your message should start from {{ and end with }}"""

#         input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
#         with torch.no_grad():
#             output = self.model.generate(
#                         input_ids,
#                         max_length=len(input_ids[0]) + 1024,
#                         min_length=len(input_ids[0]) + 10,  
#                         do_sample=True,
#                         temperature=0.7,
#                         num_return_sequences=1,
#                         no_repeat_ngram_size=2,
#                         top_k=50,
#                         top_p=0.95
#                     )


        
#         response = self.tokenizer.decode(output[0])
        
#         print(response, 'response')
#         try:
#             # Extract JSON from the response
#             json_start = response.find('{')
#             json_end = response.rfind('}') + 1
#             if json_start != -1 and json_end != -1:
#                 json_str = response[json_start:json_end]
#                 json_response = json.loads(json_str)
#                 return json_response
#             else:
#                 raise ValueError("No JSON object found in the response")
#         except json.JSONDecodeError as e:
#             print(f"JSON Decode Error: {e}")
#             print("Raw response:", response)
#             return {"error": "Failed to generate valid JSON response", "raw_response": response}
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#             return {"error": str(e), "raw_response": response}

if __name__ == "__main__":
    detector = CloneDetector()
    detector.read_directory("./mutated/")
    result = detector.detect_clones(
        """
        def func_b(potential, minister: Tuple[str, float], uncle, a):
            stat = "foo"
            if c > 2:
                stat = "foo" + str(c)
            else:
                pass
            while a > 100:
                a = math.sqrt(a)
            import math
            result = (c + a, stat)
            return result
        """
    )
    
    print(json.dumps(result, indent=2))

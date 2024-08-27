import os
import glob
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
import outlines
from typing import List, Dict, Tuple
import torch
import random
import re
import faiss
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace(' ', '_')
    return filename



class CloneReport(BaseModel):
    doc: List[str] = Field(..., description="List of file names involved in the clone")
    explanation: str = Field(..., description="Explanation of the clone relationship")

class CloneDetectionResult(BaseModel):
    clones: List[str] = Field(..., description="List of clone names")
    report: List[CloneReport] = Field(..., description="Detailed report of clone relationships")

@outlines.prompt
def clone_detection_prompt(files: List[Dict[str, str]]) -> None:
    """Analyze the following code snippets and determine if there are any clones among them:

    {% for file in files %}
    File: {{file.name}}
    Code:
    ```python
    {{file.code}}
    ```

    {% endfor %}

    Provide a list of clone names and a detailed report of clone relationships.
    For each clone relationship, provide the file names involved and an explanation.
    Output the result in the following JSON format, enclosed in ```json``` tags:

    ```json
    {
        "clones": ["file1.py", "file2.py", ...],
        "report": [
            {
                "doc": ["file1.py", "file2.py"],
                "explanation": "Explanation of the clone relationship"
            },
            ...
        ]
    }
    ```
    """


def process_directory(directory_path: str) -> List[Dict[str, str]]:
    files = glob.glob(os.path.join(directory_path, "*.py"))
    code_snippets = []
    for file in files:
        with open(file, 'r') as f:
            code_snippets.append({"name": os.path.basename(file), "code": f.read()})
    return code_snippets


import json
import re
from typing import List, Dict

def detect_clones(code_snippets: List[Dict[str, str]], outlines_model) -> CloneDetectionResult:
    prompt = clone_detection_prompt(files=code_snippets)
    generator = outlines.generate.text(outlines_model)
    print('generator in place', len(prompt))
    
    response = generator(prompt)

    print("Raw model output:")
    print(response)
    
    json_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_block_match:
        json_str = json_block_match.group(1)
        try:
            result_dict = json.loads(json_str)
            return CloneDetectionResult(**result_dict)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON within ```json``` tags. Error: {e}")
    
    # If no valid JSON found within tags, proceed with the original parsing method
    try:
        result_dict = json.loads(response)
        return CloneDetectionResult(**result_dict)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from full response. Error: {e}")
        # Attempt to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result_dict = json.loads(json_match.group())
                return CloneDetectionResult(**result_dict)
            except json.JSONDecodeError:
                print("Failed to extract valid JSON from the response.")
        
    # If all parsing attempts fail, return an empty result
    print("No valid JSON found. Returning empty result.")
    return CloneDetectionResult(clones=[], report=[])


def run_clone_detection(directory_path: str, outlines_model) -> CloneDetectionResult:
    code_snippets = process_directory(directory_path)
    return detect_clones(code_snippets, outlines_model)


def plot_clone_detection_results(result: CloneDetectionResult, model_name: str):
    num_clones = len(result.clones)
    num_files_involved = len(set([file for report in result.report for file in report.doc]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = ['Clones Detected', 'Files Involved']
    y = [num_clones, num_files_involved]
    ax.bar(x, y)
    
    ax.set_ylabel('Count')
    ax.set_title(f'Clone Detection Results - {model_name}')
    
    for i, v in enumerate(y):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create 'plots' directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Sanitize the file name
    safe_model_name = sanitize_filename(model_name)
    filepath = os.path.join('plots', f'clone_detection_results_{safe_model_name}.png')
    
    plt.savefig(filepath)
    plt.close()
    print(f"Clone detection results plot saved as: {filepath}")
    
def save_to_vector_store(result: CloneDetectionResult, code_snippets: List[Dict[str, str]], model_name: str):
    file_vectors = []
    file_names = []
    for snippet in code_snippets:
        # Use a simple hashing trick to create a vector
        vector = np.zeros(100, dtype=np.float32)
        for i, char in enumerate(snippet['code']):
            vector[hash(char) % 100] += 1
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        file_vectors.append(vector)
        file_names.append(snippet['name'])

    file_vectors = np.array(file_vectors).astype('float32')

    # Create FAISS index
    index = faiss.IndexFlatL2(100)
    index.add(file_vectors)

    os.makedirs('vector_store', exist_ok=True)

    safe_model_name = sanitize_filename(model_name)

    index_path = os.path.join('vector_store', f'faiss_index_{safe_model_name}.idx')
    faiss.write_index(index, index_path)
    print(f"FAISS index saved as: {index_path}")

    mapping_path = os.path.join('vector_store', f'file_mapping_{safe_model_name}.json')
    with open(mapping_path, 'w') as f:
        json.dump(file_names, f)
    print(f"File mapping saved as: {mapping_path}")

    # Save clone detection results
    results_path = os.path.join('vector_store', f'clone_results_{safe_model_name}.json')
    with open(results_path, 'w') as f:
        json.dump(result.dict(), f, indent=2)
    print(f"Clone detection results saved as: {results_path}")

    # Save code snippets
    snippets_path = os.path.join('vector_store', f'code_snippets_{safe_model_name}.json')
    with open(snippets_path, 'w') as f:
        json.dump(code_snippets, f, indent=2)
    print(f"Code snippets saved as: {snippets_path}")

    print(f"All vector store data saved in 'vector_store' directory with prefix: {safe_model_name}")

    return {
        'index_path': index_path,
        'mapping_path': mapping_path,
        'results_path': results_path,
        'snippets_path': snippets_path
    }

def print_results(result: CloneDetectionResult):
    print("Detected clones:")
    for clone in result.clones:
        print(f"- {clone}")
    print("\nDetailed report:")
    for report in result.report:
        print(f"Files involved: {', '.join(report.doc)}")
        print(f"Explanation: {report.explanation}")
        print("---")
        
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
hf_token = "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"

def initialize_model(model_name: str):
    model_kwargs = {
        "device_map": "cuda",
        "output_attentions": True,
    }

    # Load the model
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        **model_kwargs
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    return outlines.models.Transformers(llm, tokenizer)


def main(directory_path: str, model_name: str):
    outlines_model = initialize_model(model_name)

    code_snippets = process_directory(directory_path)
    print('\n')
    print('Prompt')

    prompt = clone_detection_prompt(files=code_snippets)
    # Run clone detection
    print("Running clone detection...")
    result = detect_clones(code_snippets, outlines_model)
    plot_clone_detection_results(result, model_name)
    
    # Save results to vector store
    save_to_vector_store(result, code_snippets, model_name)

    # Print results
    print("Clone detection results:")
    # print_results(result)
    result = []
    return result, code_snippets




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Clone Detection")
    parser.add_argument("directory_path", help="Path to the directory containing Python files")
    parser.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct", help="Name of the Hugging Face model to use")
    args = parser.parse_args()

    result, code_snippets = main(args.directory_path, args.model)

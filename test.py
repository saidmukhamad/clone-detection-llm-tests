import os
import glob
import argparse
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Union
from pydantic import BaseModel, Field
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
hf_token = "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"

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field

class CloneDetectionResult(BaseModel):
    is_clone: bool = Field(..., description="Whether the code snippets are clones or not")
    explanation: str = Field(..., description="Detailed explanation of why they are or are not clones")

class CloneReport(BaseModel):
    doc: List[str] = Field(..., description="List of file names involved in the clone")
    explanation: str = Field(..., description="Explanation of the clone relationship")



@outlines.prompt
def clone_comparison_prompt(file1: Dict[str, str], file2: Dict[str, str]) -> None:
    """Compare the following two code snippets and determine if they are clones:

    File 1: {{file1.name}}
    Code 1:
    ```python
    {{file1.code}}
    ```

    File 2: {{file2.name}}
    Code 2:
    ```python
    {{file2.code}}
    ```

    Are these code snippets clones? Provide an explanation.
    
    Strictly follow these rules when generating your response:
    1. Always use the exact JSON format provided below.
    2. The "is_clone" field must be a boolean (true or false), not a string.
    3. The "explanation" field should be a single string with newline characters (\n) for formatting.
    5. If the snippets are not clones, explain why they are fundamentally different despite any superficial similarities.
    6. If the snippets are clones, explain why they are fundamentally similar despite any superficial differences.
    7. Do not include any text outside of the JSON structure.

    Output JSON in this format:

    ```json
    {
        "is_clone": true/false,
        "explanation": "Detailed explanation of why they are or are not clones"
    }
    ```
    """


def calculate_code_complexity(code: str) -> int:
    lines = set(line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#'))
    return len(lines)

def process_directory(directory_path: str) -> List[Dict[str, str]]:
    print(f"Processing directory: {directory_path}")
    files = glob.glob(os.path.join(directory_path, "*.py"))
    print(f"Found {len(files)} Python files")

    if not files:
        raise ValueError(f"No Python files found in the directory: {directory_path}")

    code_snippets = []
    for file in files:
        with open(file, 'r') as f:
            code = f.read()
            complexity = calculate_code_complexity(code)
            if complexity > 0:
                code_snippets.append({
                    "name": os.path.basename(file),
                    "code": code,
                    "complexity": complexity
                })
                print(f"Read file: {os.path.basename(file)}, Complexity: {complexity}")
            else:
                print(f"Skipped empty or comment-only file: {os.path.basename(file)}")

    if not code_snippets:
        raise ValueError("No valid Python files found in the directory")

    print("Ranking files based on code complexity...")
    ranked_snippets = sorted(code_snippets, key=lambda x: x['complexity'], reverse=True)
    print("Files ranked successfully")
     
    return ranked_snippets

def compare_files(file1: Dict[str, str], file2: Dict[str, str], outlines_model) -> Tuple[bool, str]:
    prompt = clone_comparison_prompt(file1=file1, file2=file2)
    generator = outlines.generate.text(outlines_model)
    response = generator(prompt)
    print(response)
    result = parse_json_response(response)
    if result is None:
        print(f"Failed to parse JSON for comparison between {file1['name']} and {file2['name']}")
        return False, "Failed to parse comparison result"
    
    return result.get('is_clone', False), result.get('explanation', 'No explanation provided')

def tree_search_clones(snippets: List[Dict[str, str]], outlines_model) -> CloneDetectionResult:
    clones = []
    reports = []
    n = len(snippets)
    
    for i in range(n):
        for j in range(i+1, n):
            is_clone, explanation = compare_files(snippets[i], snippets[j], outlines_model)
            if is_clone:
                clones.extend([snippets[i]['name'], snippets[j]['name']])
                reports.append(CloneReport(
                    doc=[snippets[i]['name'], snippets[j]['name']],
                    explanation=explanation
                ))
    
    return CloneDetectionResult(clones=list(set(clones)), report=reports)

def parse_json_response(response: str) -> Union[Dict, None]:
    # First, check if the entire response is valid JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass  # If not, continue with other parsing methods
    
    # Try to extract JSON from code blocks
    json_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON within code blocks. Error: {e}")
    
    # Try to extract JSON from the response using regex
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"Failed to extract valid JSON using regex. Error: {e}")
    
    print("Failed to extract any valid JSON from the response.")
    return None


def save_json_report(result: CloneDetectionResult, directory_path: str, model_name: str):
    os.makedirs('reports', exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join('reports', f'{date_str}_report_{model_name.replace("/", "_")}.json')
    with open(filepath, 'w') as f:
        json.dump({
            "directory": directory_path,
            "model": model_name,
            "result": result.dict()
        }, f, indent=2)
    print(f"Clone detection report saved as: {filepath}")
    return filepath

def initialize_model(model_name: str):
    model_kwargs = {
        "device_map": "cuda",
        "output_attentions": True,
    }

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))

    return outlines.models.Transformers(llm, tokenizer)

def run_clone_detection(directory_path: str, model_name: str):
    outlines_model = initialize_model(model_name)
    snippets = process_directory(directory_path)
    result = tree_search_clones(snippets, outlines_model)
    json_filepath = save_json_report(result, directory_path, model_name)
    return json_filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Clone Detection")
    parser.add_argument("--directory_path", default="./mutated", help="Path to the directory containing Python files")
    parser.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct", help="Name of the Hugging Face model to use")
    args = parser.parse_args()

    json_filepath = run_clone_detection(args.directory_path, args.model)
    print(f"Clone detection completed. Results saved to: {json_filepath}")
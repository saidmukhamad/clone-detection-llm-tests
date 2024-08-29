import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

def run_clone_detection(directory_path: str, model_name: str, chunk_size: int):
    command = f"python clone_detection.py {directory_path} --model {model_name} --chunk_size {chunk_size}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return model_name, stdout.decode(), stderr.decode()

def main(directory_path: str, models: List[str], chunk_size: int, max_concurrent: int):
    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(run_clone_detection, directory_path, model, chunk_size) for model in models]
        
        for future in as_completed(futures):
            model_name, stdout, stderr = future.result()
            print(f"Results for {model_name}:")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clone Detection on Multiple Models")
    parser.add_argument("directory_path", help="Path to the directory containing Python files")
    parser.add_argument("--models", nargs="+", default=["microsoft/Phi-3.5-mini-instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"], 
                        help="List of model names to test")
    
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of files to process in each chunk")
    parser.add_argument("--max_concurrent", type=int, default=2, help="Maximum number of concurrent processes")
    args = parser.parse_args()

    main(args.directory_path, args.models, args.chunk_size, args.max_concurrent)
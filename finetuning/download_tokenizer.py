import os
import sys
import requests

def download_tokenizer(tokenizer_path="./paligemma_tokenizer.model"):
    url = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
    if not os.path.exists(tokenizer_path):
        print("Downloading the model tokenizer...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(tokenizer_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Tokenizer path: {tokenizer_path}")
        else:
            print(f"Failed to download tokenizer. Status code: {response.status_code}")
    else:
        print(f"Tokenizer already exists at: {tokenizer_path}")

def download_big_vision():
    # Check if the big_vision repository is already cloned
    if not os.path.exists("big_vision_repo"):
        print("Cloning the big_vision repository...")
        os.system("git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo")
    
    # Append big_vision code to python import path
    if "big_vision_repo" not in sys.path:
        sys.path.append("big_vision_repo")
    
    # Install missing dependencies
    print("Installing dependencies...")
    os.system("pip3 install -q overrides ml_collections einops~=0.7 sentencepiece")

if __name__ == "__main__":
    download_tokenizer()
    download_big_vision()

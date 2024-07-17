import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image

def download_model(model_id):
    print(f"Downloading model from {model_id}...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    return model, processor

def infer(model, processor, image_path, text, max_new_tokens=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image = Image.open(image_path)
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return result[0][len(text):].lstrip("\n")

def main():
    # Get model ID from user
    model_id = input("Enter the Hugging Face model ID (default: markury/paligemma-448-ft-1): ") or "markury/paligemma-448-ft-1"

    # Download model
    model, processor = download_model(model_id)

    while True:
        # Get image path and text input from user
        image_path = input("Enter the path to the image file: ")
        text = input("Enter the input text: ")

        # Perform inference
        result = infer(model, processor, image_path, text)

        print("\nOutput:")
        print(result)

        # Ask if the user wants to continue
        if input("\nDo you want to process another image? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()
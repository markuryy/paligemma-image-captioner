import argparse
import os
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
from tqdm import tqdm

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    if os.path.isdir(model_path):
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
        processor = PaliGemmaProcessor.from_pretrained(model_path)
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
        processor = PaliGemmaProcessor.from_pretrained(model_path)
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

def process_single_image(model, processor, image_path, text, output_file=None):
    result = infer(model, processor, image_path, text)
    print(f"\nCaption for {image_path}:")
    print(result)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"Caption saved to {output_file}")

def process_batch(model, processor, folder_path, text):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(folder_path, image_file)
        result = infer(model, processor, image_path, text)
        
        output_file = os.path.splitext(image_path)[0] + '.txt'
        with open(output_file, 'w') as f:
            f.write(result)
    
    print(f"\nProcessed {len(image_files)} images. Captions saved as text files in the same directory.")

def main():
    parser = argparse.ArgumentParser(description="PaliGemma Image Captioner CLI")
    parser.add_argument("--model", required=True, help="Path to local model or Hugging Face model ID")
    parser.add_argument("--mode", choices=["single", "batch"], required=True, help="Single image or batch processing mode")
    parser.add_argument("--image", help="Path to the image file (for single mode)")
    parser.add_argument("--folder", help="Path to the folder containing images (for batch mode)")
    parser.add_argument("--text", default="", help="Input text for captioning")
    parser.add_argument("--output", help="Output file path for single mode caption (optional)")

    args = parser.parse_args()

    model, processor = load_model(args.model)

    if args.mode == "single":
        if not args.image:
            parser.error("--image is required when using single mode")
        process_single_image(model, processor, args.image, args.text, args.output)
    elif args.mode == "batch":
        if not args.folder:
            parser.error("--folder is required when using batch mode")
        process_batch(model, processor, args.folder, args.text)

if __name__ == "__main__":
    main()
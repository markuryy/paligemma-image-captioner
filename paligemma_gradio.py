import os
import torch
import gradio as gr
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import logging
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaliGemmaCaption:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.should_stop = False

    def load_model(self, model_path):
        try:
            logging.info(f"Loading model from {model_path}")
            if model_path.lower().endswith('.npz'):
                raise ValueError("NPZ files are not supported. Please convert to safetensors using the provided Colab notebook.")
            
            if os.path.isdir(model_path):
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
                self.processor = PaliGemmaProcessor.from_pretrained(model_path)
            else:
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
                self.processor = PaliGemmaProcessor.from_pretrained(model_path)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logging.info("Model loaded successfully")
            return "Model loaded successfully"
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return f"Error loading model: {str(e)}"

    def infer(self, image, text, max_new_tokens=128):
        if self.model is None or self.processor is None:
            return "Please load a model first."

        try:
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return result[0][len(text):].lstrip("\n")
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}")
            return f"Error during inference: {str(e)}"

    def process_batch(self, folder_path, text, max_new_tokens, progress=gr.Progress()):
        if self.model is None or self.processor is None:
            return "Please load a model first."

        try:
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            results = []

            for image_file in progress.tqdm(image_files):
                if self.should_stop:
                    logging.info("Batch processing interrupted by user")
                    break
                
                image_path = os.path.join(folder_path, image_file)
                image = Image.open(image_path)
                caption = self.infer(image, text, max_new_tokens)
                results.append((image_path, image, caption))

                # Save caption as text file
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                with open(txt_path, 'w') as f:
                    f.write(caption)

            self.should_stop = False
            return results
        except Exception as e:
            logging.error(f"Error during batch processing: {str(e)}")
            return f"Error during batch processing: {str(e)}"

    def get_system_info(self):
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                return f"CPU: {cpu_percent}% | RAM: {memory.percent}% | GPU: {gpu_memory:.2f} GB"
            else:
                return f"CPU: {cpu_percent}% | RAM: {memory.percent}% | GPU: N/A"
        except Exception as e:
            logging.error(f"Error getting system info: {str(e)}")
            return f"Error getting system info: {str(e)}"

pali_gemma = PaliGemmaCaption()

def load_model_gradio(model_path):
    result = pali_gemma.load_model(model_path)
    return result, pali_gemma.get_system_info()

def infer_gradio(image, text, max_tokens):
    caption = pali_gemma.infer(image, text, max_tokens)
    return caption, pali_gemma.get_system_info()

def process_batch_gradio(folder_path, text, max_tokens, progress=gr.Progress()):
    results = pali_gemma.process_batch(folder_path, text, max_tokens, progress)
    if isinstance(results, str):  # Error occurred
        return None, results
    
    table_data = [[os.path.basename(path), caption] for path, _, caption in results]
    return table_data, pali_gemma.get_system_info()

def interrupt_batch():
    pali_gemma.should_stop = True
    return "Interrupting batch processing..."

title = """<center><img src="https://www.markury.dev/logo.png" style="width: 75px; height: 75px; margin-top: 10px; margin-bottom: 10px;"></center>
<h1 align="center">M/PIC</h1>
<p align="center">Markury's PaliGemma Image Captioner</p>
<p><center>
<a href="https://github.com/markuryy/paligemma-image-captioner" target="_blank">[Github]</a>
<a href="https://huggingface.co/markury/AndroGemma-alpha" target="_blank">[Models]</a>
</center></p>
"""
def create_gradio_interface():
    with gr.Blocks(theme='bethecloud/storj_theme') as demo:
        gr.HTML(title)
                
        with gr.Row():
            with gr.Column(scale=3):
                model_path = gr.Textbox(label="Model Path or Hugging Face ID")
                model_suggestions = gr.Dropdown(
                    choices=["google/paligemma-3b-mix-448", "markury/AndroGemma-alpha"],
                    label="Preset models"
                )
                load_button = gr.Button("Load Model")
            with gr.Column(scale=2):
                info_output = gr.Textbox(label="Info Console", lines=3)
        
        model_suggestions.change(lambda x: x, inputs=[model_suggestions], outputs=[model_path])
        load_button.click(load_model_gradio, inputs=[model_path], outputs=[info_output, info_output])
        
        with gr.Tabs():
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Input Image")
                        text_input = gr.Textbox(label="Input Text", value="caption en")
                        max_tokens = gr.Slider(minimum=1, maximum=512, value=128, step=1, label="Max New Tokens")
                        caption_button = gr.Button("Generate Caption")
                    with gr.Column():
                        caption_output = gr.Textbox(label="Generated Caption")
                
                caption_button.click(infer_gradio, inputs=[image_input, text_input, max_tokens], outputs=[caption_output, info_output])
            
            with gr.TabItem("Batch Processing"):
                folder_input = gr.Textbox(label="Folder Path")
                with gr.Row():
                    batch_text_input = gr.Textbox(label="Input Text", value="caption en")
                    batch_max_tokens = gr.Slider(minimum=1, maximum=512, value=128, step=1, label="Max New Tokens")
                with gr.Row():
                    batch_button = gr.Button("Process Batch")
                    interrupt_button = gr.Button("Interrupt")
                
                batch_output = gr.DataFrame(
                    headers=["File Name", "Caption"],
                    datatype=["str", "str"],
                    label="Batch Results"
                )
                
                batch_button.click(process_batch_gradio, inputs=[folder_input, batch_text_input, batch_max_tokens], outputs=[batch_output, info_output])
                interrupt_button.click(interrupt_batch, outputs=info_output)
        
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()

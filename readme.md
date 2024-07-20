# PaliGemma Image Captioner

This repository contains a GUI application for image captioning using your own finetune of the PaliGemma model. The application allows for both single image and batch processing, with options to use local models or models from Hugging Face.

## Features

- Load PaliGemma models from local directories or Hugging Face
- Single image captioning with preview
- Batch processing of images in a folder
- Edit and save generated captions
- Modern and user-friendly interface

## Requirements

- Python 3.7 or higher
- PyQt6
- torch
- transformers
- Pillow

## Installation

### Setting up a Virtual Environment

It's recommended to use a virtual environment for this project. Here's how to set it up on different operating systems:

#### Windows

```
python -m venv venv
venv\Scripts\activate
```

#### macOS and Linux

```
python3 -m venv venv
source venv/bin/activate
```

### Installing Dependencies

Once your virtual environment is activated, install the required packages:

```
pip install -r requirements.txt
```

## Usage

1. Activate your virtual environment (if not already activated).

2. Run the application:

```
python paligemma_gui.py
```

3. Load a PaliGemma model:
   - Enter a local path to a PaliGemma model directory, or
   - Enter a Hugging Face model ID (e.g., "markury/paligemma-448-ft-1")
   - Click "Load Model"

4. For single image captioning:
   - Go to the "Single Image" tab
   - Select an image or enter an image path
   - (Optional) Enter input text
   - Click "Generate Caption"

5. For batch processing:
   - Go to the "Batch Processing" tab
   - Select a folder containing images
   - (Optional) Enter input text for batch processing
   - Click "Process Batch"
   - Double-click on results to edit captions

## Troubleshooting

- If you encounter any CUDA-related errors, ensure that your PyTorch installation matches your CUDA version.
- For "module not found" errors, make sure you've activated the virtual environment and installed all dependencies.
- If the GUI doesn't launch, check that you have PyQt6 installed correctly.

## Using the Command-Line Interface

For scenarios where a graphical user interface is not available (e.g., running on a remote server or in a cloud environment), you can use the command-line interface (CLI) version of the PaliGemma Image Captioner.

### CLI Script Usage

The CLI script (`paligemma_cli.py`) supports both single image captioning and batch processing. Here's how to use it:

1. Ensure you have all the required dependencies installed:

```
pip install torch transformers pillow tqdm
```

2. Run the script with the appropriate arguments:

```
python paligemma_cli.py --model <model_path_or_id> --mode <single|batch> [other options]
```

#### Arguments:

- `--model`: Path to local model or Hugging Face model ID (required)
- `--mode`: Choose between "single" for single image processing or "batch" for processing a folder of images (required)
- `--image`: Path to the image file (required for single mode)
- `--folder`: Path to the folder containing images (required for batch mode)
- `--text`: Input text for captioning (optional)
- `--output`: Output file path for single mode caption (optional)

#### Examples:

1. Single image captioning:

```
python paligemma_cli.py --model markury/paligemma-448-ft-1 --mode single --image path/to/image.jpg --text "Describe this image:" --output caption.txt
```

2. Batch processing:

```
python paligemma_cli.py --model path/to/local/model --mode batch --folder path/to/image/folder --text "describe"
```

### Note:

- For batch processing, the script will automatically save captions as text files with the same name as the image files in the specified folder.
- The script uses CUDA if available, otherwise it falls back to CPU processing.
- Progress is displayed using a progress bar for batch processing.

Remember to activate your virtual environment before running the script if you're using one.

## Contributing

Contributions to improve the PaliGemma Image Captioner are welcome. Please feel free to submit pull requests or create issues for bugs and feature requests.

## Acknowledgements

This project uses the PaliGemma model, which is based on the work by Google Research. Special thanks to the Hugging Face team for providing easy access to transformer models.
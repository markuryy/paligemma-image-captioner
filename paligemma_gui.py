import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QTextEdit, QLabel, QFileDialog, QProgressBar, 
                             QListWidget, QMessageBox, QInputDialog, QSizePolicy, QComboBox, QScrollArea)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image

class InferenceWorker(QObject):
    progress = pyqtSignal(int)
    result = pyqtSignal(str, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, processor, image_paths, text):
        super().__init__()
        self.model = model
        self.processor = processor
        self.image_paths = image_paths
        self.text = text

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            self.model.eval()

            for i, image_path in enumerate(self.image_paths):
                image = Image.open(image_path)
                inputs = self.processor(text=self.text, images=image, return_tensors="pt").to(device)

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False
                    )
                
                result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                output = result[0][len(self.text):].lstrip("\n")

                self.result.emit(image_path, output)
                self.progress.emit(int((i + 1) / len(self.image_paths) * 100))

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class PaliGemmaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Markury's PaliGemma Image Captioner")
        self.setGeometry(100, 100, 1000, 800)

        self.model = None
        self.processor = None
        self.thread = None
        self.worker = None

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Model loading section
        model_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Enter local model path or Hugging Face model ID")
        model_layout.addWidget(self.model_path_input)
        
        # Add model suggestions
        self.model_suggestions = QComboBox()
        self.model_suggestions.addItem("Preset models")
        self.model_suggestions.addItem("google/paligemma-3b-mix-448")
        self.model_suggestions.addItem("markury/paligemma-448-ft-1")
        self.model_suggestions.currentIndexChanged.connect(self.update_model_input)
        model_layout.addWidget(self.model_suggestions)
        
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        main_layout.addLayout(model_layout)

        # Loading indicator
        self.loading_label = QLabel("Loading...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setVisible(False)
        main_layout.addWidget(self.loading_label)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_single_image_tab(), "Single Image")
        self.tabs.addTab(self.create_batch_processing_tab(), "Batch Processing")
        main_layout.addWidget(self.tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_single_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Image selection
        image_layout = QHBoxLayout()
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("Enter image path or select an image")
        image_layout.addWidget(self.image_path_input)
        select_image_btn = QPushButton("Select Image")
        select_image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(select_image_btn)
        layout.addLayout(image_layout)

        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.image_preview)

        # Text input
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter input text (optional)")
        layout.addWidget(self.text_input)

        # Generate button
        generate_btn = QPushButton("Generate Caption")
        generate_btn.clicked.connect(self.generate_single_caption)
        layout.addWidget(generate_btn)

        # Output
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        tab.setLayout(layout)
        return tab

    def create_batch_processing_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_path_input = QLineEdit()
        self.folder_path_input.setPlaceholderText("Enter folder path")
        folder_layout.addWidget(self.folder_path_input)
        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(select_folder_btn)
        layout.addLayout(folder_layout)

        # Text input for batch processing
        self.batch_text_input = QLineEdit()
        self.batch_text_input.setPlaceholderText("Enter input text for batch processing (optional)")
        layout.addWidget(self.batch_text_input)

        # Process button
        process_btn = QPushButton("Process Batch")
        process_btn.clicked.connect(self.process_batch)
        layout.addWidget(process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Results list and image preview
        results_layout = QHBoxLayout()
        
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.show_batch_image_preview)
        self.results_list.itemDoubleClicked.connect(self.edit_caption)
        results_layout.addWidget(self.results_list, 2)
        
        self.batch_image_preview = QLabel()
        self.batch_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.batch_image_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        results_layout.addWidget(self.batch_image_preview, 1)
        
        layout.addLayout(results_layout)

        tab.setLayout(layout)
        return tab

    def update_model_input(self, index):
        if index > 0:
            self.model_path_input.setText(self.model_suggestions.currentText())

    def load_model(self):
        model_path = self.model_path_input.text()
        self.loading_label.setVisible(True)
        QApplication.processEvents()

        try:
            if model_path.lower().endswith('.npz'):
                raise ValueError("NPZ files are not supported. Please convert to safetensors using the provided Colab notebook.")
            
            if os.path.isdir(model_path):
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
                self.processor = PaliGemmaProcessor.from_pretrained(model_path)
            else:
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
                self.processor = PaliGemmaProcessor.from_pretrained(model_path)
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        finally:
            self.loading_label.setVisible(False)

    def select_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if image_path:
            self.image_path_input.setText(image_path)
            self.update_image_preview(image_path, self.image_preview)

    def update_image_preview(self, image_path, preview_label):
        pixmap = QPixmap(image_path)
        preview_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

    def generate_single_caption(self):
        if not self.model or not self.processor:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return

        image_path = self.image_path_input.text()
        text = self.text_input.text()

        if not image_path:
            QMessageBox.warning(self, "Warning", "Please provide an image path.")
            return

        self.loading_label.setVisible(True)
        QApplication.processEvents()

        self.thread = QThread()
        self.worker = InferenceWorker(self.model, self.processor, [image_path], text)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: self.loading_label.setVisible(False))
        
        self.worker.result.connect(self.update_single_result)
        self.worker.error.connect(self.show_error)
        
        self.thread.start()

    def update_single_result(self, image_path, output):
        self.output_text.setText(output)

    def select_folder(self):
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path_input.setText(folder_path)

    def process_batch(self):
        if not self.model or not self.processor:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return

        folder_path = self.folder_path_input.text()
        text = self.batch_text_input.text()

        if not folder_path:
            QMessageBox.warning(self, "Warning", "Please provide a folder path.")
            return

        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.loading_label.setVisible(True)
        QApplication.processEvents()

        self.thread = QThread()
        self.worker = InferenceWorker(self.model, self.processor, image_paths, text)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: self.loading_label.setVisible(False))
        
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.update_batch_result)
        self.worker.error.connect(self.show_error)
        
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_batch_result(self, image_path, output):
        item = f"{os.path.basename(image_path)}: {output}"
        self.results_list.addItem(item)

        # Save to text file
        txt_path = os.path.splitext(image_path)[0] + '.txt'
        with open(txt_path, 'w') as f:
            f.write(output)

    def show_batch_image_preview(self, item):
        image_name = item.text().split(':')[0].strip()
        folder_path = self.folder_path_input.text()
        image_path = os.path.join(folder_path, image_name)
        self.update_image_preview(image_path, self.batch_image_preview)

    def edit_caption(self, item):
        old_text = item.text()
        new_text, ok = QInputDialog.getText(self, "Edit Caption", "Enter new caption:", QLineEdit.EchoMode.Normal, old_text)
        if ok and new_text:
            item.setText(new_text)
            # Update the corresponding text file
            image_name = old_text.split(':')[0].strip()
            folder_path = self.folder_path_input.text()
            txt_path = os.path.join(folder_path, os.path.splitext(image_name)[0] + '.txt')
            with open(txt_path, 'w') as f:
                f.write(new_text.split(': ', 1)[1])

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.loading_label.setVisible(False)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit, QTextEdit, QListWidget {
                border: 1px solid #cccccc;
                padding: 6px;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QLabel#loading_label {
                font-weight: bold;
                color: #4CAF50;
            }
        """)

        # Set a modern font
        font = QFont("Segoe UI", 10)
        QApplication.setFont(font)

    def closeEvent(self, event):
        # Ensure that any running threads are properly closed
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PaliGemmaGUI()
    window.show()
    sys.exit(app.exec())
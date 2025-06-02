# Advanced Text Extraction Software

A comprehensive, AI-powered text extraction solution capable of extracting text from images and documents with high accuracy and advanced processing capabilities.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

## 🚀 Features

### 🔍 Multi-Engine OCR
- **Tesseract OCR**: Industry-standard OCR with 100+ language support
- **EasyOCR**: Deep learning-based OCR with neural network models
- **PaddleOCR**: Baidu's production-ready OCR system
- **Intelligent Engine Selection**: Automatically chooses the best engine for each task

### 📄 Document Processing
- **PDF Files**: Multiple extraction methods (PyPDF2, pdfplumber, PyMuPDF)
- **Microsoft Office**: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- **Legacy Excel**: Support for .xls files
- **Plain Text**: Direct text file processing
- **Batch Processing**: Handle multiple files simultaneously

### 🖼️ Advanced Image Processing
- **Noise Reduction**: Remove artifacts and improve clarity
- **Deskewing**: Automatically correct tilted text
- **Contrast Enhancement**: Optimize text visibility
- **Binarization**: Convert to optimal black/white format
- **Shadow Removal**: Eliminate scanning shadows
- **Preprocessing Pipeline**: Intelligent image optimization

### 🧠 AI-Powered Features
- **Language Detection**: Automatic language identification
- **Confidence Scoring**: Quality assessment for extracted text
- **Text Quality Analysis**: Evaluate extraction accuracy
- **Metadata Extraction**: File information and processing details
- **Smart Error Correction**: Improve OCR accuracy

### 💾 Data Management
- **SQLite Database**: Store extraction history and cache results
- **Multiple Export Formats**: JSON, CSV, TXT
- **Progress Tracking**: Real-time processing status
- **History Management**: Track all extraction activities

### 🖥️ User Interfaces
- **Modern GUI**: Dark-themed interface with drag-and-drop support
- **Command Line**: Powerful CLI with Rich console output
- **Unified Launcher**: Single entry point with automatic setup

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space for dependencies

### Tesseract OCR
Install Tesseract OCR for your operating system:

#### Windows
```bash
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Or using chocolatey:
choco install tesseract
```

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/bonin1/advanced_text_extraction.git
cd advanced_text_extraction
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Setup Check
```bash
python main.py --check-deps
```

## 🚀 Quick Start

### GUI Interface
Launch the graphical interface:
```bash
python main.py --gui
```

**Features:**
- Drag and drop files for instant processing
- Real-time preview and editing
- Multiple export options
- Dark modern theme
- Progress tracking

### Command Line Interface
Run from command line:
```bash
python main.py --cli
```

**Basic Usage:**
```bash
# Extract text from an image
python cli.py extract image.png

# Process a PDF document
python cli.py extract document.pdf --engine tesseract

# Batch process multiple files
python cli.py batch-extract folder/ --output results/

# Export in different formats
python cli.py extract file.jpg --export json,csv,txt
```

### Python API
Use in your own projects:
```python
from text_extractor import AdvancedTextExtractor

# Initialize the extractor
extractor = AdvancedTextExtractor()

# Extract text from an image
result = extractor.extract_from_file("image.png")
print(f"Extracted text: {result.text}")
print(f"Confidence: {result.confidence}%")

# Extract from PDF
pdf_result = extractor.extract_from_file("document.pdf")
print(f"Pages processed: {len(pdf_result.metadata.get('pages', []))}")

# Batch processing
files = ["file1.png", "file2.pdf", "file3.docx"]
results = extractor.batch_extract(files)
```

## 📖 Usage Examples

### Image Text Extraction
```python
# Simple image extraction
extractor = AdvancedTextExtractor()
result = extractor.extract_from_file("receipt.jpg")

# With preprocessing
result = extractor.extract_from_file(
    "low_quality_scan.png",
    preprocess=True,
    enhance_contrast=True,
    remove_noise=True
)

# Multiple engines for comparison
result = extractor.extract_text_multi_engine("document.png")
for engine, text in result.items():
    print(f"{engine}: {text[:100]}...")
```

### Document Processing
```python
# PDF extraction with multiple methods
pdf_results = extractor.extract_from_pdf("report.pdf", method="all")

# Word document
doc_result = extractor.extract_from_file("document.docx")

# Excel file
excel_result = extractor.extract_from_file("spreadsheet.xlsx")
```

### Batch Processing
```python
import os

# Process entire directory
input_dir = "documents/"
output_dir = "extracted_text/"

files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
results = extractor.batch_extract(files)

# Save results
for i, result in enumerate(results):
    output_file = os.path.join(output_dir, f"result_{i}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result.text)
```

### Advanced Features
```python
# Language detection
result = extractor.extract_from_file("french_document.jpg")
print(f"Detected language: {result.language}")

# Quality assessment
if result.confidence < 70:
    print("Low confidence - consider preprocessing")

# Metadata extraction
print(f"Processing time: {result.metadata['processing_time']}")
print(f"File size: {result.metadata['file_size']}")
```

## 📊 Export Formats

### JSON Export
```json
{
    "text": "Extracted text content...",
    "confidence": 85.7,
    "language": "en",
    "metadata": {
        "filename": "document.pdf",
        "file_size": 1024000,
        "processing_time": 2.34,
        "engine_used": "tesseract",
        "pages_processed": 3
    },
    "extraction_date": "2024-01-15T10:30:00Z"
}
```

### CSV Export
Ideal for spreadsheet analysis and batch processing results.

### TXT Export
Clean text output for further processing or reading.

## ⚙️ Configuration

### Engine Selection
```python
# Force specific engine
extractor = AdvancedTextExtractor(preferred_engine="easyocr")

# Engine priority
extractor.set_engine_priority(["paddleocr", "easyocr", "tesseract"])
```

### Image Preprocessing
```python
# Custom preprocessing pipeline
preprocessing_config = {
    "denoise": True,
    "deskew": True,
    "enhance_contrast": True,
    "binarize": True,
    "remove_shadows": True
}

result = extractor.extract_from_file(
    "image.jpg",
    preprocessing=preprocessing_config
)
```

### Language Configuration
```python
# Specific language for Tesseract
extractor.set_tesseract_language("eng+fra+deu")  # English, French, German

# Language detection settings
extractor.enable_language_detection(confidence_threshold=0.8)
```

## 🧪 Testing

### Run Tests
```bash
# Basic functionality test
python demo.py

# Comprehensive test
python final_demo.py

# Individual component tests
python test_modules.py
```

### Benchmark Performance
```python
import time

# Performance testing
start_time = time.time()
result = extractor.extract_from_file("large_document.pdf")
processing_time = time.time() - start_time

print(f"Processing time: {processing_time:.2f} seconds")
print(f"Characters per second: {len(result.text) / processing_time:.2f}")
```

## 🔧 Troubleshooting

### Common Issues

#### Tesseract Not Found
```bash
# Add Tesseract to PATH or specify location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### Low OCR Accuracy
- Ensure good image quality (300+ DPI recommended)
- Use preprocessing options
- Try different OCR engines
- Specify correct language

#### Memory Issues
- Process files individually instead of batch
- Reduce image resolution for large files
- Close extractor instance when done

#### Dependencies
```bash
# Check missing dependencies
python main.py --check-deps

# Install missing packages
pip install -r requirements.txt --upgrade
```

### Performance Optimization

#### For Large Files
```python
# Process PDF pages individually
for page_num in range(pdf_pages):
    page_result = extractor.extract_from_pdf_page("large.pdf", page_num)
```

#### For Batch Processing
```python
# Use multiprocessing for CPU-intensive tasks
extractor.set_max_workers(4)  # Adjust based on CPU cores
```

## 🤝 Contributing

### Development Setup
```bash
git clone https://github.com/yourusername/advanced_text_extraction.git
cd advanced_text_extraction
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Write unit tests for new features

### Submitting Changes
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📈 Performance Benchmarks

### OCR Engine Comparison
| Engine | Speed | Accuracy | Language Support | Memory Usage |
|--------|-------|----------|------------------|--------------|
| Tesseract | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| EasyOCR | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| PaddleOCR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### File Format Support
- **Images**: PNG, JPG, JPEG, BMP, TIFF, GIF
- **Documents**: PDF, DOCX, XLSX, PPTX, TXT, XLS
- **Web**: HTML (via URL processing)

## 🔒 Security

### Data Privacy
- All processing is done locally
- No data sent to external servers (except for translation features)
- Temporary files are automatically cleaned up
- Extraction history can be disabled

### Safe Processing
- Input validation for all file types
- Memory limits to prevent system overload
- Error handling for malformed files

## 🙏 Acknowledgments

- **Tesseract OCR** - Google's OCR engine
- **EasyOCR** - JaidedAI's deep learning OCR
- **PaddleOCR** - Baidu's production OCR system
- **OpenCV** - Computer vision library
- **CustomTkinter** - Modern GUI framework

### Getting Help
- 📖 Check this README for common solutions
- 🐛 Report bugs via GitHub Issues
- 💬 Ask questions in GitHub Discussions

### Useful Resources
- [Tesseract Documentation](https://tesseract-ocr.github.io/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [PIL/Pillow Documentation](https://pillow.readthedocs.io/)

---

**Made with ❤️ for the text extraction community**

*Star ⭐ this repository if you find it helpful!*

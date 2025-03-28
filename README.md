# Advanced Data Extractor

A powerful Flask-based web application for extracting structured data from various document formats including PDFs, Word documents, Excel spreadsheets, and images with advanced OCR capabilities.

## Features

- **Multi-format support**: Process PDFs, Word documents, Excel spreadsheets, and images
- **Advanced OCR**: Extract text from scanned documents and images with Tesseract OCR
- **Table extraction**: Automatically identify and extract tables from documents
- **Layout preservation**: Maintain the original document layout during text extraction
- **Metadata extraction**: Extract document properties and metadata
- **Font analysis**: Identify and extract font information from PDFs
- **Document structure analysis**: Identify headers, sections, and keywords using NLP
- **Form field extraction**: Extract form fields from PDFs
- **Multi-language support**: Process documents in multiple languages

## Installation

### Prerequisites

1. **Python 3.7+**
   - Download and install Python from [python.org](https://python.org)

2. **Tesseract OCR**
   - **Windows**: Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: Install using Homebrew: `brew install tesseract`
   - **Linux**: Install using your package manager: `sudo apt-get install tesseract-ocr`

3. **Required Python Libraries**
   - Core libraries for basic functionality
   - Advanced libraries for enhanced features

### Setup Instructions

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required Python packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up spaCy for NLP capabilities**
   ```bash
   python setup_nlp.py
   ```

5. **Configure Tesseract OCR**
   - Ensure Tesseract is in your system PATH
   - For Windows users: Add the Tesseract installation directory (e.g., `C:\Program Files\Tesseract-OCR`) to your PATH

## Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

## Usage Guide

### Uploading Documents

1. Click **Browse Files** or drag and drop your document into the upload area
2. Select the file you want to process

### Configuring Extraction Options

#### Basic Options
- **OCR Language**: Choose the language for OCR (for images and scanned documents)
- **OCR Mode**: Select the quality level for OCR processing
- **Tesseract Path**: Specify the path to your Tesseract executable if it's not detected automatically
- **Extract Tables**: Enable table detection and extraction
- **Preserve Layout**: Maintain the original document layout during extraction

#### Advanced Options
- **DPI for Image Processing**: Set the resolution for image conversion (higher values provide better quality but slower processing)
- **Processing Threads**: Set the number of parallel processing threads
- **Image Preprocessing**: Apply image enhancement before OCR
- **Font Analysis**: Extract font information from PDFs
- **Extract Form Fields**: Identify and extract form fields from PDFs

### Viewing Results

After processing, you'll see:
- **Processing Summary**: Statistics about the processed document
- **Extracted Data Preview**: Structured data extracted from the document, including:
  - Metadata
  - Keywords and entities
  - Tables
  - Font information
  - Form fields
- **Extracted Files**: Download or preview the extracted text and structured data

## Troubleshooting

### Tesseract OCR Issues

If the application can't find Tesseract OCR:

1. Verify your Tesseract installation by running `tesseract --version` in a terminal/command prompt
2. Make sure Tesseract is in your system PATH or specify the full path in the application
3. Check the Tesseract installation guide for more details: [Tesseract Installation Guide](http://localhost:5000/tesseract_instructions)

### SpaCy Model Issues

If you encounter issues with the spaCy language model:

1. Run the NLP setup script: `python setup_nlp.py`
2. Manually install the spaCy model: `python -m spacy download en_core_web_sm`

### Other Issues

- Check that all required libraries are installed: `pip install -r requirements.txt`
- Ensure you have appropriate permissions to read/write files
- For large files, increase the timeout and memory limits in the Flask configuration

## Advanced Configuration

### Environment Variables

- `SECRET_KEY`: Set a custom secret key for Flask sessions
- Set using: `export SECRET_KEY=your_secret_key` (Linux/macOS) or `set SECRET_KEY=your_secret_key` (Windows)

### Command Line Usage

The extraction tool can also be used from the command line:

```bash
python main.py path/to/your/file.pdf --ocr-lang eng --ocr-mode advanced --tesseract-path "/path/to/tesseract"
```

Common options:
- `--ocr-lang`: Language for OCR (e.g., eng, fra, deu)
- `--output-dir`: Directory to save extracted data
- `--dpi`: DPI for image conversion (higher = better quality but slower)
- `--ocr-mode`: OCR processing mode (basic, advanced, deep)
- `--tesseract-path`: Path to Tesseract executable
- `--threads`: Maximum number of threads for parallel processing

## Dependencies

### Core Libraries
- Flask: Web framework
- PyPDF2: Basic PDF processing
- Pandas: Data handling and Excel processing
- Pillow (PIL): Image processing
- pytesseract: OCR interface for Tesseract
- pdf2image: Convert PDFs to images

### Advanced Libraries
- pdfplumber: Enhanced PDF text extraction
- pikepdf: Advanced PDF manipulation
- tabula-py: Table extraction from PDFs
- camelot-py: Enhanced table extraction
- python-docx: Word document processing
- spaCy: Natural language processing
- nltk: Text processing and analysis
- OpenCV (cv2): Advanced image preprocessing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [spaCy](https://spacy.io/) - Industrial-strength NLP
- [Camelot](https://camelot-py.readthedocs.io/) - PDF Table Extraction
- [Tabula](https://tabula-py.readthedocs.io/) - PDF Table Extraction
- [pdfplumber](https://github.com/jsvine/pdfplumber) - Plumb PDFs for detailed information

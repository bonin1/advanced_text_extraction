from flask import Flask, render_template, request, send_file, redirect, url_for, flash, jsonify, session
import os
import uuid
from werkzeug.utils import secure_filename
from main import DataExtractor, download_spacy_model, find_tesseract_executable
import logging
import json
import time
from pathlib import Path
import sys

# Initialize and check for spaCy model before starting the app
try:
    download_spacy_model("en_core_web_sm")
except Exception as e:
    logging.warning(f"Could not download spaCy model: {e}. Some NLP features may be limited.")

# Try to find Tesseract executable
tesseract_path = find_tesseract_executable()
if tesseract_path:
    logging.info(f"Found Tesseract at: {tesseract_path}")
else:
    logging.warning("Tesseract not found in common locations. OCR features may be limited.")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'extracted_data')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size
app.config['TESSERACT_PATH'] = tesseract_path or ''

# Create upload and output folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'xlsx', 'xls', 'csv', 'docx', 'doc', 
    'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif'
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the home page with the upload form"""
    return render_template('index.html', tesseract_path=app.config['TESSERACT_PATH'])

@app.route('/extract', methods=['POST'])
def extract_file():
    """Handle file upload and extraction with advanced options"""
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    # Get form options
    ocr_lang = request.form.get('ocr_lang', 'eng')
    extract_tables = 'extract_tables' in request.form
    preserve_layout = 'preserve_layout' in request.form
    preprocess = 'preprocess' in request.form
    ocr_mode = request.form.get('ocr_mode', 'advanced')
    dpi = int(request.form.get('dpi', '300'))
    font_analysis = 'font_analysis' in request.form
    extract_forms = 'extract_forms' in request.form
    max_threads = int(request.form.get('threads', '4'))
    tesseract_path = request.form.get('tesseract_path', app.config['TESSERACT_PATH'])
    
    # Save the tesseract path in app config if it's provided and valid
    if tesseract_path and os.path.isfile(tesseract_path):
        app.config['TESSERACT_PATH'] = tesseract_path
    
    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        try:
            # Initialize the extractor with advanced options
            extractor = DataExtractor(
                ocr_lang=ocr_lang,
                extract_tables=extract_tables,
                output_dir=app.config['OUTPUT_FOLDER'],
                dpi=dpi,
                preprocess=preprocess,
                preserve_layout=preserve_layout,
                ocr_mode=ocr_mode,
                font_analysis=font_analysis,
                extract_forms=extract_forms,
                max_threads=max_threads,
                tesseract_path=tesseract_path
            )
            
            # Extract data from the file
            start_time = time.time()
            result = extractor.extract(file_path)
            processing_time = time.time() - start_time
            
            if result and not result.get("error"):
                # Store processing stats in session
                stats = {
                    'filename': file.filename,
                    'processing_time': f"{processing_time:.2f}",
                    'file_type': result.get('type', 'unknown'),
                    'pages_processed': len(result.get('text', [])),
                    'tables_extracted': len(result.get('tables', [])),
                }
                
                # Store stats in a JSON file for retrieval on results page
                stats_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{Path(file_path).stem}_stats.json")
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f)
                
                # Redirect to the result page
                return redirect(url_for('show_result', filename=unique_filename))
            else:
                # Handle extraction error
                error_msg = result.get("error", "Unknown error during extraction")
                flash(f"Extraction failed: {error_msg}")
                return redirect(url_for('index'))
                
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            flash(f"Error processing file: {str(e)}")
            return redirect(url_for('index'))
    else:
        flash(f'Unsupported file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}')
        return redirect(url_for('index'))

@app.route('/results/<filename>')
def show_result(filename):
    """Show extraction results with enhanced metadata"""
    # Get the original filename without unique prefix
    original_filename = filename.split('_', 1)[1] if '_' in filename else filename
    base_name = Path(filename).stem
    
    # Load processing stats if available
    stats = {}
    stats_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        except:
            pass
    
    # Find all extracted files related to this upload
    base_stem = Path(base_name).stem
    extracted_files = []
    structured_data = None
    
    # Look for extracted files
    for file in os.listdir(app.config['OUTPUT_FOLDER']):
        if base_stem in file:
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], file)
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file)[1][1:].lower()  # Get extension without dot
            
            file_info = {
                'name': file,
                'path': file_path,
                'size': file_size,
                'type': file_ext.upper(),
                'display_size': format_file_size(file_size)
            }
            
            # Try to load structured data for preview
            if file_ext == 'json' and 'structured' in file:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        structured_data = json.load(f)
                    file_info['is_structured_data'] = True
                except:
                    file_info['is_structured_data'] = False
                    
            extracted_files.append(file_info)
    
    # Sort files by type (put text files first)
    extracted_files.sort(key=lambda x: 0 if x['type'] == 'TXT' else 1)
    
    return render_template(
        'results.html', 
        filename=original_filename,
        extracted_files=extracted_files,
        stats=stats,
        structured_data=structured_data
    )

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download an extracted file"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(
            file_path, 
            as_attachment=True, 
            download_name=filename
        )
    else:
        flash('File not found')
        return redirect(url_for('index'))

@app.route('/preview/<path:filename>')
def preview_file(filename):
    """Preview a text file's contents"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path) and filename.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({'content': content})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'File not found or not a text file'})

@app.route('/tesseract_instructions')
def tesseract_instructions():
    """Show instructions for installing Tesseract OCR"""
    detected_path = find_tesseract_executable()
    return render_template('tesseract_instructions.html', detected_path=detected_path)

def format_file_size(size_bytes):
    """Format file size in bytes to human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.1f} GB"

if __name__ == '__main__':
    app.run(debug=True)
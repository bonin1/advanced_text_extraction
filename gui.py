"""
Advanced Text Extraction GUI
===========================

Modern dark-themed GUI interface for the Advanced Text Extraction Software
using CustomTkinter with drag-and-drop support, batch processing, and
real-time preview capabilities.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import webbrowser

from text_extractor import AdvancedTextExtractor, ExtractionResult


class ProgressDialog(ctk.CTkToplevel):
    """Progress dialog for batch processing"""
    
    def __init__(self, parent, title="Processing"):
        super().__init__(parent)
        
        self.title(title)
        self.geometry("400x200")
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.winfo_screenheight() // 2) - (200 // 2)
        self.geometry(f"400x200+{x}+{y}")
        
        # Create widgets
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.status_label = ctk.CTkLabel(
            self.main_frame, 
            text="Preparing...",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=(10, 20))
        
        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_bar.set(0)
        
        self.progress_text = ctk.CTkLabel(
            self.main_frame, 
            text="0 / 0 files processed",
            font=ctk.CTkFont(size=12)
        )
        self.progress_text.pack(pady=(0, 20))
        
        self.cancel_button = ctk.CTkButton(
            self.main_frame,
            text="Cancel",
            command=self.cancel_processing
        )
        self.cancel_button.pack()
        
        self.cancelled = False
        
    def update_progress(self, current: int, total: int, current_file: str = ""):
        """Update progress display"""
        if self.cancelled:
            return
            
        progress = current / total if total > 0 else 0
        self.progress_bar.set(progress)
        
        self.progress_text.configure(text=f"{current} / {total} files processed")
        
        if current_file:
            filename = Path(current_file).name
            if len(filename) > 40:
                filename = filename[:37] + "..."
            self.status_label.configure(text=f"Processing: {filename}")
        
        self.update()
    
    def cancel_processing(self):
        """Cancel the processing"""
        self.cancelled = True
        self.destroy()


class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog for configuration"""
    
    def __init__(self, parent, config: dict):
        super().__init__(parent)
        
        self.title("Settings")
        self.geometry("500x600")
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.winfo_screenheight() // 2) - (600 // 2)
        self.geometry(f"500x600+{x}+{y}")
        
        self.config = config.copy()
        self.result = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create settings widgets"""
        # Main frame with scrollbar
        self.main_frame = ctk.CTkScrollableFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # OCR Settings
        ocr_frame = ctk.CTkFrame(self.main_frame)
        ocr_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(ocr_frame, text="OCR Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # Default OCR Engine
        engine_frame = ctk.CTkFrame(ocr_frame)
        engine_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(engine_frame, text="Default OCR Engine:").pack(side="left", padx=10)
        self.engine_var = ctk.StringVar(value=self.config.get('ocr', {}).get('default_engine', 'auto'))
        engine_menu = ctk.CTkOptionMenu(
            engine_frame,
            variable=self.engine_var,
            values=["auto", "tesseract", "easyocr", "paddleocr"]
        )
        engine_menu.pack(side="right", padx=10, pady=10)
        
        # Preprocessing
        preprocess_frame = ctk.CTkFrame(ocr_frame)
        preprocess_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(preprocess_frame, text="Enable Preprocessing:").pack(side="left", padx=10)
        self.preprocess_var = ctk.BooleanVar(value=self.config.get('ocr', {}).get('preprocess', True))
        preprocess_switch = ctk.CTkSwitch(preprocess_frame, variable=self.preprocess_var)
        preprocess_switch.pack(side="right", padx=10, pady=10)
        
        # Confidence Threshold
        conf_frame = ctk.CTkFrame(ocr_frame)
        conf_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        ctk.CTkLabel(conf_frame, text="Confidence Threshold:").pack(side="left", padx=10)
        self.confidence_var = ctk.DoubleVar(value=self.config.get('ocr', {}).get('confidence_threshold', 0.5))
        conf_slider = ctk.CTkSlider(conf_frame, from_=0.0, to=1.0, variable=self.confidence_var)
        conf_slider.pack(side="right", padx=10, pady=10)
        
        # Batch Processing Settings
        batch_frame = ctk.CTkFrame(self.main_frame)
        batch_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(batch_frame, text="Batch Processing", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # Max Workers
        workers_frame = ctk.CTkFrame(batch_frame)
        workers_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(workers_frame, text="Max Workers:").pack(side="left", padx=10)
        self.workers_var = ctk.IntVar(value=self.config.get('batch', {}).get('max_workers', 4))
        workers_entry = ctk.CTkEntry(workers_frame, textvariable=self.workers_var, width=100)
        workers_entry.pack(side="right", padx=10, pady=10)
          # Database Settings
        db_frame = ctk.CTkFrame(self.main_frame)
        db_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(db_frame, text="Database Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # Database Path
        path_frame = ctk.CTkFrame(db_frame)
        path_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        ctk.CTkLabel(path_frame, text="Database Path:").pack(anchor="w", padx=10, pady=(10, 5))
        self.db_path_var = ctk.StringVar(value=self.config.get('database', {}).get('path', 'extraction_history.db'))
        path_entry = ctk.CTkEntry(path_frame, textvariable=self.db_path_var)
        path_entry.pack(fill="x", padx=10, pady=(0, 10))
        
        # Save Settings
        save_frame = ctk.CTkFrame(self.main_frame)
        save_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(save_frame, text="Save & Auto-Save Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # Auto-save enabled
        autosave_frame = ctk.CTkFrame(save_frame)
        autosave_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(autosave_frame, text="Enable Auto-Save:").pack(side="left", padx=10)
        self.autosave_var = ctk.BooleanVar(value=self.config.get('save', {}).get('auto_save_enabled', False))
        autosave_switch = ctk.CTkSwitch(autosave_frame, variable=self.autosave_var)
        autosave_switch.pack(side="right", padx=10, pady=10)
        
        # Default save directory
        save_dir_frame = ctk.CTkFrame(save_frame)
        save_dir_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(save_dir_frame, text="Default Save Directory:").pack(anchor="w", padx=10, pady=(10, 5))
        self.save_dir_var = ctk.StringVar(value=self.config.get('save', {}).get('default_directory', './output'))
        
        save_dir_entry_frame = ctk.CTkFrame(save_dir_frame, fg_color="transparent")
        save_dir_entry_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        save_dir_entry = ctk.CTkEntry(save_dir_entry_frame, textvariable=self.save_dir_var)
        save_dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ctk.CTkButton(
            save_dir_entry_frame,
            text="Browse",
            command=self.browse_save_directory,
            width=80
        ).pack(side="right")
        
        # Save format preference
        format_frame = ctk.CTkFrame(save_frame)
        format_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(format_frame, text="Default Save Format:").pack(side="left", padx=10)
        self.save_format_var = ctk.StringVar(value=self.config.get('save', {}).get('default_format', 'txt'))
        format_menu = ctk.CTkOptionMenu(
            format_frame,
            variable=self.save_format_var,
            values=["txt", "json", "csv"]
        )
        format_menu.pack(side="right", padx=10, pady=10)
        
        # Auto-save after extraction
        auto_after_frame = ctk.CTkFrame(save_frame)
        auto_after_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        ctk.CTkLabel(auto_after_frame, text="Auto-save after extraction:").pack(side="left", padx=10)
        self.auto_after_var = ctk.BooleanVar(value=self.config.get('save', {}).get('auto_save_after_extraction', False))
        auto_after_switch = ctk.CTkSwitch(auto_after_frame, variable=self.auto_after_var)
        auto_after_switch.pack(side="right", padx=10, pady=10)
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="Save",
            command=self.save_settings
        ).pack(side="right", padx=(10, 0))
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side="right")
        
        ctk.CTkButton(
            button_frame,
            text="Reset to Defaults",
            command=self.reset_defaults
        ).pack(side="left")
    
    def browse_save_directory(self):
        """Browse for save directory"""
        directory = filedialog.askdirectory(
            title="Select Default Save Directory",
            initialdir=self.save_dir_var.get()
        )
        if directory:
            self.save_dir_var.set(directory)
    def save_settings(self):
        """Save current settings"""
        self.result = {
            'ocr': {
                'default_engine': self.engine_var.get(),
                'preprocess': self.preprocess_var.get(),
                'confidence_threshold': self.confidence_var.get()
            },
            'batch': {
                'max_workers': self.workers_var.get()
            },
            'database': {
                'path': self.db_path_var.get()
            },
            'save': {
                'auto_save_enabled': self.autosave_var.get(),
                'default_directory': self.save_dir_var.get(),
                'default_format': self.save_format_var.get(),
                'auto_save_after_extraction': self.auto_after_var.get()
            }
        }
        self.destroy()
    def reset_defaults(self):
        """Reset to default settings"""
        self.engine_var.set('auto')
        self.preprocess_var.set(True)
        self.confidence_var.set(0.5)
        self.workers_var.set(4)
        self.db_path_var.set('extraction_history.db')
        self.autosave_var.set(False)
        self.save_dir_var.set('./output')
        self.save_format_var.set('txt')
        self.auto_after_var.set(False)


class SaveDialog(ctk.CTkToplevel):
    """Dialog for advanced save options with format selection"""
    
    def __init__(self, parent, results: List[ExtractionResult], save_settings: dict):
        super().__init__(parent)
        
        self.title("Save Results - Advanced Options")
        self.geometry("550x400")
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (550 // 2)
        y = (self.winfo_screenheight() // 2) - (400 // 2)
        self.geometry(f"550x400+{x}+{y}")
        
        self.results = results
        self.save_settings = save_settings
        self.saved_files = []
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create save dialog widgets"""
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="💾 Advanced Save Options",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Info
        info_label = ctk.CTkLabel(
            main_frame,
            text=f"Save {len(self.results)} extraction results in multiple formats",
            font=ctk.CTkFont(size=12)
        )
        info_label.pack(pady=(0, 20))
        
        # Save directory selection
        dir_frame = ctk.CTkFrame(main_frame)
        dir_frame.pack(fill="x", padx=10, pady=(0, 15))
        
        ctk.CTkLabel(dir_frame, text="Save Directory:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        dir_entry_frame = ctk.CTkFrame(dir_frame, fg_color="transparent")
        dir_entry_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.dir_var = ctk.StringVar(value=self.save_settings.get('default_directory', './output'))
        dir_entry = ctk.CTkEntry(dir_entry_frame, textvariable=self.dir_var)
        dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ctk.CTkButton(
            dir_entry_frame,
            text="Browse",
            command=self.browse_directory,
            width=80
        ).pack(side="right")
        
        # Format selection
        format_frame = ctk.CTkFrame(main_frame)
        format_frame.pack(fill="x", padx=10, pady=(0, 15))
        
        ctk.CTkLabel(format_frame, text="Select Formats to Save:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Format checkboxes
        checkbox_frame = ctk.CTkFrame(format_frame, fg_color="transparent")
        checkbox_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.format_vars = {}
        default_format = self.save_settings.get('default_format', 'txt')
        
        formats = [
            ('txt', '📄 Text File (.txt)', default_format == 'txt'),
            ('json', '🔧 JSON File (.json)', default_format == 'json'),
            ('csv', '📊 CSV File (.csv)', default_format == 'csv')
        ]
        
        for fmt, label, default in formats:
            var = ctk.BooleanVar(value=default)
            self.format_vars[fmt] = var
            
            checkbox = ctk.CTkCheckBox(
                checkbox_frame,
                text=label,
                variable=var,
                font=ctk.CTkFont(size=12)
            )
            checkbox.pack(anchor="w", padx=10, pady=2)
        
        # Save options
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.pack(fill="x", padx=10, pady=(0, 15))
        
        ctk.CTkLabel(options_frame, text="Save Options:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Individual files option
        self.individual_var = ctk.BooleanVar(value=True)
        individual_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="Save individual files (one per source file)",
            variable=self.individual_var,
            font=ctk.CTkFont(size=12)
        )
        individual_checkbox.pack(anchor="w", padx=10, pady=2)
        
        # Combined file option
        self.combined_var = ctk.BooleanVar(value=True)
        combined_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="Save combined file (all results together)",
            variable=self.combined_var,
            font=ctk.CTkFont(size=12)
        )
        combined_checkbox.pack(anchor="w", padx=10, pady=(2, 10))
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy,
            width=100
        ).pack(side="right", padx=(10, 0))
        
        ctk.CTkButton(
            button_frame,
            text="Save All",
            command=self.save_files,
            width=100,
            fg_color="#1f8b4c",
            hover_color="#2d7d32"
        ).pack(side="right")
    
    def browse_directory(self):
        """Browse for save directory"""
        directory = filedialog.askdirectory(
            title="Select Save Directory",
            initialdir=self.dir_var.get()
        )
        if directory:
            self.dir_var.set(directory)
    
    def save_files(self):
        """Save files with selected options"""
        # Validate selections
        selected_formats = [fmt for fmt, var in self.format_vars.items() if var.get()]
        if not selected_formats:
            messagebox.showwarning("No Format Selected", "Please select at least one format to save.")
            return
        
        if not (self.individual_var.get() or self.combined_var.get()):
            messagebox.showwarning("No Save Option Selected", "Please select at least one save option (individual or combined).")
            return
        
        try:
            save_dir = self.dir_var.get()
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for format_type in selected_formats:
                if self.individual_var.get():
                    # Save individual files
                    individual_dir = os.path.join(save_dir, f"individual_{format_type}")
                    os.makedirs(individual_dir, exist_ok=True)
                    
                    for i, result in enumerate(self.results):
                        source_name = Path(result.source_file).stem
                        filename = f"{source_name}_extracted_{timestamp}.{format_type}"
                        filepath = os.path.join(individual_dir, filename)
                        
                        # Save single result
                        self._save_single_result(result, filepath, format_type)
                        self.saved_files.append(filepath)
                
                if self.combined_var.get():
                    # Save combined file
                    combined_filename = f"combined_results_{timestamp}.{format_type}"
                    combined_filepath = os.path.join(save_dir, combined_filename)
                    
                    # Use extractor's export method for combined results
                    from text_extractor import AdvancedTextExtractor
                    extractor = AdvancedTextExtractor()
                    extractor.export_results(self.results, combined_filepath, format_type)
                    self.saved_files.append(combined_filepath)
            
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Save Failed", f"Failed to save files:\n{str(e)}")
    
    def _save_single_result(self, result: ExtractionResult, filepath: str, format_type: str):
        """Save a single result to file"""
        if format_type == 'json':
            output_data = {
                'source_file': result.source_file,
                'text': result.text,
                'confidence': result.confidence,
                'language': result.language,
                'extraction_method': result.extraction_method,
                'processing_time': result.processing_time,
                'timestamp': result.timestamp.isoformat(),
                'metadata': result.metadata
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'csv':
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Source File', 'Text', 'Confidence', 'Language', 'Method', 'Processing Time', 'Timestamp'])
                writer.writerow([
                    result.source_file,
                    result.text.replace('\n', ' '),
                    result.confidence,
                    result.language,
                    result.extraction_method,
                    result.processing_time,
                    result.timestamp.isoformat()
                ])
        
        else:  # txt format
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {result.source_file}\n")
                f.write(f"Language: {result.language}\n")
                f.write(f"Confidence: {result.confidence:.2f}\n")
                f.write(f"Method: {result.extraction_method}\n")
                f.write(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
                f.write(result.text)


class TextExtractionGUI:
    """Main GUI application class"""
    
    def __init__(self):
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize main window
        self.root = TkinterDnD.Tk()
        self.root.title("Advanced Text Extraction Software")
        self.root.geometry("1200x800")
          # Initialize extractor
        self.extractor = AdvancedTextExtractor()
        self.current_results = []
        
        # Configuration and save settings
        self.app_config = self.load_app_config()
        self.save_settings = self.app_config.get('save', {})
        
        # File handling
        self.selected_files = []
        self.processing_thread = None
        
        self.create_widgets()
        self.setup_drag_and_drop()
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """Center the main window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        # Create header
        self.create_header()
        
        # Create left panel (file selection and controls)
        self.create_left_panel()
        
        # Create right panel (results and preview)
        self.create_right_panel()
        
        # Create bottom status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create application header"""
        header_frame = ctk.CTkFrame(self.main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🚀 Advanced Text Extraction Software",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=15)
        
        # Header buttons
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.pack(side="right", padx=20, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="⚙️ Settings",
            width=100,
            command=self.open_settings
        ).pack(side="right", padx=(10, 0))
        
        ctk.CTkButton(
            button_frame,
            text="📊 History",
            width=100,
            command=self.show_history
        ).pack(side="right", padx=(10, 0))
        
        ctk.CTkButton(
            button_frame,
            text="❓ Help",
            width=80,
            command=self.show_help
        ).pack(side="right")
    
    def create_left_panel(self):
        """Create left panel with file selection and controls"""
        left_frame = ctk.CTkFrame(self.main_frame)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        left_frame.grid_rowconfigure(2, weight=1)
        
        # File selection section
        file_section = ctk.CTkFrame(left_frame)
        file_section.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            file_section,
            text="📁 File Selection",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # File buttons
        button_frame = ctk.CTkFrame(file_section, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkButton(
            button_frame,
            text="Add Files",
            command=self.add_files,
            width=120
        ).pack(side="left", padx=(0, 5))
        
        ctk.CTkButton(
            button_frame,
            text="Add Folder",
            command=self.add_folder,
            width=120
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=self.clear_files,
            width=100
        ).pack(side="right")
        
        # Drag and drop area
        self.drop_frame = ctk.CTkFrame(left_frame)
        self.drop_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        self.drop_label = ctk.CTkLabel(
            self.drop_frame,
            text="🎯 Drag & Drop Files Here\\n\\nSupported formats:\\n• Images: JPG, PNG, BMP, TIFF\\n• Documents: PDF, DOCX, XLSX, PPTX\\n• Text: TXT",
            font=ctk.CTkFont(size=14),
            justify="center"
        )
        self.drop_label.pack(expand=True, fill="both", padx=20, pady=30)
        
        # File list
        list_frame = ctk.CTkFrame(left_frame)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        list_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(
            list_frame,
            text="📋 Selected Files",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # File listbox with scrollbar
        listbox_frame = ctk.CTkFrame(list_frame)
        listbox_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        listbox_frame.grid_rowconfigure(0, weight=1)
        listbox_frame.grid_columnconfigure(0, weight=1)
        
        self.file_listbox = tk.Listbox(
            listbox_frame,
            bg="#212121",
            fg="white",
            selectbackground="#1f538d",
            font=("Consolas", 10)
        )
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        
        listbox_scrollbar = ctk.CTkScrollbar(listbox_frame, command=self.file_listbox.yview)
        listbox_scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_listbox.configure(yscrollcommand=listbox_scrollbar.set)
        
        # Processing controls
        control_frame = ctk.CTkFrame(left_frame)
        control_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        
        ctk.CTkLabel(
            control_frame,
            text="🚀 Processing",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # Processing options
        options_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        options_frame.pack(fill="x", padx=10)
        
        ctk.CTkLabel(options_frame, text="OCR Engine:").grid(row=0, column=0, sticky="w", pady=5)
        self.engine_var = ctk.StringVar(value="auto")
        engine_menu = ctk.CTkOptionMenu(
            options_frame,
            variable=self.engine_var,
            values=["auto", "tesseract", "easyocr", "paddleocr"],
            width=120
        )
        engine_menu.grid(row=0, column=1, sticky="e", pady=5)
        
        ctk.CTkLabel(options_frame, text="Preprocessing:").grid(row=1, column=0, sticky="w", pady=5)
        self.preprocess_var = ctk.BooleanVar(value=True)
        preprocess_switch = ctk.CTkSwitch(options_frame, variable=self.preprocess_var)
        preprocess_switch.grid(row=1, column=1, sticky="e", pady=5)
        
        options_frame.grid_columnconfigure(1, weight=1)
        
        # Process button
        self.process_button = ctk.CTkButton(
            control_frame,
            text="🚀 Start Extraction",
            command=self.start_processing,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.process_button.pack(fill="x", padx=10, pady=10)
    
    def create_right_panel(self):
        """Create right panel with results and preview"""
        right_frame = ctk.CTkFrame(self.main_frame)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
        right_frame.grid_rowconfigure(1, weight=1)
        
        # Results header
        results_header = ctk.CTkFrame(right_frame)
        results_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            results_header,
            text="📄 Extraction Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left", padx=10, pady=10)
          # Export and Save buttons
        export_frame = ctk.CTkFrame(results_header, fg_color="transparent")
        export_frame.pack(side="right", padx=10)
        
        # Save buttons
        ctk.CTkButton(
            export_frame,
            text="💾 Save All",
            command=self.save_all_results,
            width=100,
            fg_color="#1f8b4c",
            hover_color="#2d7d32"
        ).pack(side="right", padx=(5, 0))
        
        ctk.CTkButton(
            export_frame,
            text="💾 Quick Save",
            command=self.quick_save_results,
            width=100,
            fg_color="#1976d2",
            hover_color="#1565c0"
        ).pack(side="right", padx=5)
        
        # Export buttons
        ctk.CTkButton(
            export_frame,
            text="📤 Export JSON",
            command=lambda: self.export_results('json'),
            width=110
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            export_frame,
            text="📤 Export CSV",
            command=lambda: self.export_results('csv'),
            width=110
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            export_frame,
            text="📤 Export TXT",
            command=lambda: self.export_results('txt'),
            width=110
        ).pack(side="right")
        
        # Results notebook (tabs)
        self.results_notebook = ctk.CTkTabview(right_frame)
        self.results_notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Preview tab
        self.results_notebook.add("Preview")
        preview_frame = self.results_notebook.tab("Preview")
        
        self.preview_text = ctk.CTkTextbox(
            preview_frame,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.preview_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Summary tab
        self.results_notebook.add("Summary")
        summary_frame = self.results_notebook.tab("Summary")
        
        self.summary_text = ctk.CTkTextbox(
            summary_frame,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Details tab
        self.results_notebook.add("Details")
        details_frame = self.results_notebook.tab("Details")
        
        self.details_text = ctk.CTkTextbox(
            details_frame,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.details_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_status_bar(self):
        """Create bottom status bar"""
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=20, pady=10)
        
        self.file_count_label = ctk.CTkLabel(
            self.status_frame,
            text="0 files selected",
            font=ctk.CTkFont(size=12)
        )
        self.file_count_label.pack(side="right", padx=20, pady=10)
    
    def setup_drag_and_drop(self):
        """Setup drag and drop functionality"""
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)
    
    def on_drop(self, event):
        """Handle dropped files"""
        files = self.root.tk.splitlist(event.data)
        self.add_files_to_list(files)
    
    def add_files(self):
        """Open file dialog to add files"""
        file_types = [
            ("All Supported", "*.pdf;*.docx;*.xlsx;*.pptx;*.txt;*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif;*.webp"),
            ("Documents", "*.pdf;*.docx;*.xlsx;*.pptx;*.txt"),
            ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif;*.webp"),
            ("All Files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select files to extract text from",
            filetypes=file_types
        )
        
        if files:
            self.add_files_to_list(files)
    
    def add_folder(self):
        """Add all supported files from a folder"""
        folder = filedialog.askdirectory(title="Select folder")
        
        if folder:
            supported_extensions = {
                '.pdf', '.docx', '.xlsx', '.pptx', '.txt',
                '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
            }
            
            files = []
            for root, dirs, filenames in os.walk(folder):
                for filename in filenames:
                    if Path(filename).suffix.lower() in supported_extensions:
                        files.append(os.path.join(root, filename))
            
            if files:
                self.add_files_to_list(files)
                self.update_status(f"Added {len(files)} files from folder")
            else:
                messagebox.showinfo("No Files", "No supported files found in the selected folder.")
    
    def add_files_to_list(self, files):
        """Add files to the selection list"""
        added_count = 0
        
        for file_path in files:
            if file_path not in self.selected_files:
                self.selected_files.append(file_path)
                self.file_listbox.insert(tk.END, Path(file_path).name)
                added_count += 1
        
        self.update_file_count()
        
        if added_count > 0:
            self.update_status(f"Added {added_count} files")
    
    def clear_files(self):
        """Clear all selected files"""
        self.selected_files.clear()
        self.file_listbox.delete(0, tk.END)
        self.update_file_count()
        self.update_status("Files cleared")
    
    def update_file_count(self):
        """Update file count display"""
        count = len(self.selected_files)
        self.file_count_label.configure(text=f"{count} file{'s' if count != 1 else ''} selected")
    
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_label.configure(text=message)
        self.root.update_idletasks()
    
    def start_processing(self):
        """Start text extraction in background thread"""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select files to process.")
            return
        
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Processing", "Processing is already in progress.")
            return
        
        # Disable process button
        self.process_button.configure(state="disabled", text="Processing...")
        
        # Clear previous results
        self.current_results.clear()
        self.clear_results_display()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_files_thread,
            daemon=True
        )
        self.processing_thread.start()
    
    def process_files_thread(self):
        """Process files in background thread"""
        try:
            # Create progress dialog
            progress_dialog = ProgressDialog(self.root, "Extracting Text")
            
            def progress_callback(current, total, current_file):
                if not progress_dialog.cancelled:
                    self.root.after(0, lambda: progress_dialog.update_progress(current, total, current_file))
                else:
                    # Cancel processing
                    return False
                return True
            
            # Process files
            results = []
            for i, file_path in enumerate(self.selected_files):
                if progress_dialog.cancelled:
                    break
                
                try:
                    result = self.extractor.extract_from_file(file_path)
                    results.append(result)
                    
                    # Update progress
                    if not progress_callback(i + 1, len(self.selected_files), file_path):
                        break
                        
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            # Close progress dialog
            self.root.after(0, progress_dialog.destroy)
            
            if not progress_dialog.cancelled and results:
                self.current_results = results
                self.root.after(0, self.display_results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:            # Re-enable process button
            self.root.after(0, lambda: self.process_button.configure(
                state="normal", 
                text="🚀 Start Extraction"
            ))
    
    def display_results(self):
        """Display extraction results in the GUI"""
        if not self.current_results:
            return
        
        # Update preview tab
        self.update_preview_tab()
        
        # Update summary tab
        self.update_summary_tab()
        
        # Update details tab
        self.update_details_tab()
        
        # Switch to preview tab
        self.results_notebook.set("Preview")
        
        # Auto-save if enabled
        self.auto_save_after_extraction()
        
        self.update_status(f"Extraction completed - {len(self.current_results)} files processed")
    
    def update_preview_tab(self):
        """Update the preview tab with extracted text"""
        self.preview_text.delete("0.0", tk.END)
        
        if not self.current_results:
            self.preview_text.insert("0.0", "No results to display.")
            return
        
        preview_content = []
        
        for i, result in enumerate(self.current_results, 1):
            filename = Path(result.source_file).name
            preview_content.append(f"=== File {i}: {filename} ===")
            preview_content.append(f"Language: {result.language}")
            preview_content.append(f"Confidence: {result.confidence:.2f}")
            preview_content.append(f"Method: {result.extraction_method}")
            preview_content.append("-" * 50)
            
            # Limit preview text length
            text_preview = result.text[:1000]
            if len(result.text) > 1000:
                text_preview += "\\n\\n[Text truncated...]"
            
            preview_content.append(text_preview)
            preview_content.append("\\n" + "=" * 70 + "\\n")
        
        self.preview_text.insert("0.0", "\\n".join(preview_content))
    
    def update_summary_tab(self):
        """Update the summary tab with statistics"""
        self.summary_text.delete("0.0", tk.END)
        
        if not self.current_results:
            self.summary_text.insert("0.0", "No results to summarize.")
            return
        
        # Calculate statistics
        total_files = len(self.current_results)
        total_chars = sum(len(r.text) for r in self.current_results)
        total_words = sum(len(r.text.split()) for r in self.current_results)
        avg_confidence = sum(r.confidence for r in self.current_results) / total_files
        total_time = sum(r.processing_time for r in self.current_results)
        
        # Language distribution
        languages = {}
        for result in self.current_results:
            lang = result.language
            languages[lang] = languages.get(lang, 0) + 1
        
        # Method distribution
        methods = {}
        for result in self.current_results:
            method = result.extraction_method
            methods[method] = methods.get(method, 0) + 1
        
        # Generate summary
        summary_lines = [
            "📊 EXTRACTION SUMMARY",
            "=" * 50,
            f"Total Files Processed: {total_files}",
            f"Total Characters Extracted: {total_chars:,}",
            f"Total Words Extracted: {total_words:,}",
            f"Average Confidence: {avg_confidence:.2f}",
            f"Total Processing Time: {total_time:.2f} seconds",
            f"Average Time per File: {total_time/total_files:.2f} seconds",
            "",
            "📈 LANGUAGE DISTRIBUTION:",
            "-" * 30
        ]
        
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_files) * 100
            summary_lines.append(f"{lang}: {count} files ({percentage:.1f}%)")
        
        summary_lines.extend([
            "",
            "🔧 METHOD DISTRIBUTION:",
            "-" * 30
        ])
        
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_files) * 100
            summary_lines.append(f"{method}: {count} files ({percentage:.1f}%)")
        
        summary_lines.extend([
            "",
            "🎯 CONFIDENCE DISTRIBUTION:",
            "-" * 30
        ])
        
        # Confidence ranges
        high_conf = sum(1 for r in self.current_results if r.confidence >= 0.8)
        med_conf = sum(1 for r in self.current_results if 0.5 <= r.confidence < 0.8)
        low_conf = sum(1 for r in self.current_results if r.confidence < 0.5)
        
        summary_lines.extend([
            f"High (≥0.8): {high_conf} files ({high_conf/total_files*100:.1f}%)",
            f"Medium (0.5-0.8): {med_conf} files ({med_conf/total_files*100:.1f}%)",
            f"Low (<0.5): {low_conf} files ({low_conf/total_files*100:.1f}%)"
        ])
        
        self.summary_text.insert("0.0", "\\n".join(summary_lines))
    
    def update_details_tab(self):
        """Update the details tab with per-file information"""
        self.details_text.delete("0.0", tk.END)
        
        if not self.current_results:
            self.details_text.insert("0.0", "No details to display.")
            return
        
        details_lines = [
            "📋 DETAILED RESULTS",
            "=" * 80
        ]
        
        for i, result in enumerate(self.current_results, 1):
            filename = Path(result.source_file).name
            file_size = result.metadata.get('file_size', 0)
            
            details_lines.extend([
                f"",
                f"File {i}: {filename}",
                f"Path: {result.source_file}",
                f"Size: {file_size:,} bytes" if file_size else "Size: Unknown",
                f"Method: {result.extraction_method}",
                f"Language: {result.language}",
                f"Confidence: {result.confidence:.3f}",
                f"Processing Time: {result.processing_time:.3f} seconds",
                f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Characters: {len(result.text):,}",
                f"Words: {len(result.text.split()):,}",
                f"Lines: {result.text.count(chr(10)) + 1:,}",
                "-" * 80
            ])
        
        self.details_text.insert("0.0", "\\n".join(details_lines))
    
    def clear_results_display(self):
        """Clear all results displays"""
        self.preview_text.delete("0.0", tk.END)
        self.summary_text.delete("0.0", tk.END)
        self.details_text.delete("0.0", tk.END)
    
    def export_results(self, format_type: str):
        """Export results to file"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No results to export.")
            return
          # File dialog
        file_types = {
            'json': [("JSON files", "*.json")],
            'csv': [("CSV files", "*.csv")],
            'txt': [("Text files", "*.txt")]
        }
        
        filename = filedialog.asksaveasfilename(
            title=f"Export as {format_type.upper()}",
            defaultextension=f".{format_type}",
            filetypes=file_types[format_type] + [("All files", "*.*")]
        )
        
        if filename:
            try:
                self.extractor.export_results(self.current_results, filename, format_type)
                messagebox.showinfo("Export Successful", f"Results exported to {filename}")
                self.update_status(f"Results exported to {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export results: {str(e)}")
    
    def open_settings(self):
        """Open settings dialog"""
        # Merge extractor config with app config for the dialog
        merged_config = self.extractor.config.copy()
        merged_config.update(self.app_config)
        
        settings_dialog = SettingsDialog(self.root, merged_config)
        self.root.wait_window(settings_dialog)
        
        if settings_dialog.result:
            # Update extractor configuration
            extractor_config = {
                'ocr': settings_dialog.result.get('ocr', {}),
                'batch': settings_dialog.result.get('batch', {}),
                'database': settings_dialog.result.get('database', {})
            }
            self.extractor.config.update(extractor_config)
            
            # Update app configuration
            if 'save' in settings_dialog.result:
                self.app_config['save'] = settings_dialog.result['save']
                self.save_settings = self.app_config['save']
                self.save_app_config()
            
            self.update_status("Settings updated")
    
    def show_history(self):
        """Show extraction history"""
        history = self.extractor.get_extraction_history(50)
        
        if not history:
            messagebox.showinfo("History", "No extraction history found.")
            return
        
        # Create history window
        history_window = ctk.CTkToplevel(self.root)
        history_window.title("Extraction History")
        history_window.geometry("800x600")
        
        # Create treeview for history
        frame = ctk.CTkFrame(history_window)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Use tkinter Treeview for tabular data
        tree_frame = tk.Frame(frame)
        tree_frame.pack(fill="both", expand=True)
        
        columns = ("File", "Method", "Language", "Confidence", "Time", "Date")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        tree.heading("File", text="File")
        tree.heading("Method", text="Method")
        tree.heading("Language", text="Language")
        tree.heading("Confidence", text="Confidence")
        tree.heading("Time", text="Time (s)")
        tree.heading("Date", text="Date")
        
        tree.column("File", width=200)
        tree.column("Method", width=100)
        tree.column("Language", width=80)
        tree.column("Confidence", width=100)
        tree.column("Time", width=80)
        tree.column("Date", width=150)
        
        # Add data
        for entry in history:
            filename = Path(entry['file_path']).name
            tree.insert("", "end", values=(
                filename,
                entry['extraction_method'],
                entry['language'],
                f"{entry['confidence']:.2f}",
                f"{entry['processing_time']:.2f}",
                entry['timestamp']
            ))
        
        tree.pack(side="left", fill="both", expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
    
    def show_help(self):
        """Show help information"""
        help_text = """
🚀 Advanced Text Extraction Software - Help

SUPPORTED FILE TYPES:
• Images: JPG, JPEG, PNG, BMP, TIFF, WEBP
• Documents: PDF, DOCX, XLSX, PPTX, TXT

HOW TO USE:
1. Add files by clicking "Add Files" or "Add Folder"
2. Or drag and drop files directly onto the interface
3. Select OCR engine and preprocessing options
4. Click "Start Extraction" to begin processing
5. View results in the Preview, Summary, or Details tabs
6. Export results as JSON, CSV, or TXT

OCR ENGINES:
• Auto: Automatically selects the best engine
• Tesseract: Industry standard OCR
• EasyOCR: AI-powered with 80+ languages
• PaddleOCR: High accuracy for Asian languages

FEATURES:
• Batch processing with progress tracking
• Intelligent image preprocessing
• Language detection and confidence scoring
• Extraction history with caching
• Multiple export formats

For more information, visit the project repository.
        """
        
        messagebox.showinfo("Help", help_text)
    
    def load_app_config(self):
        """Load application configuration from file"""
        config_file = "app_config.json"
        default_config = {
            'save': {
                'auto_save_enabled': False,
                'default_directory': './output',
                'default_format': 'txt',
                'auto_save_after_extraction': False
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def save_app_config(self):
        """Save application configuration to file"""
        config_file = "app_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.app_config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def quick_save_results(self):
        """Quick save results using default settings"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No results to save.")
            return
        
        try:
            # Create default save directory if it doesn't exist
            save_dir = self.save_settings.get('default_directory', './output')
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            format_type = self.save_settings.get('default_format', 'txt')
            filename = f"extraction_results_{timestamp}.{format_type}"
            filepath = os.path.join(save_dir, filename)
            
            # Save the results
            self.extractor.export_results(self.current_results, filepath, format_type)
            
            messagebox.showinfo("Save Successful", f"Results saved to:\n{filepath}")
            self.update_status(f"Quick saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Save Failed", f"Failed to save results:\n{str(e)}")
    
    def save_all_results(self):
        """Save results with format selection dialog"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No results to save.")
            return
        
        # Show save dialog
        save_dialog = SaveDialog(self.root, self.current_results, self.save_settings)
        self.root.wait_window(save_dialog)
        
        if save_dialog.saved_files:
            files_str = '\n'.join(save_dialog.saved_files)
            messagebox.showinfo("Save Successful", f"Results saved to:\n{files_str}")
            self.update_status(f"Saved {len(save_dialog.saved_files)} files")
    
    def auto_save_after_extraction(self):
        """Auto-save results after extraction if enabled"""
        if not self.save_settings.get('auto_save_after_extraction', False):
            return
        
        if not self.current_results:
            return
        
        try:
            # Create auto-save directory
            save_dir = os.path.join(self.save_settings.get('default_directory', './output'), 'auto_saves')
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            format_type = self.save_settings.get('default_format', 'txt')
            filename = f"auto_save_{timestamp}.{format_type}"
            filepath = os.path.join(save_dir, filename)
            
            # Save the results
            self.extractor.export_results(self.current_results, filepath, format_type)
            
            self.update_status(f"Auto-saved to {filename}")
            
        except Exception as e:
            print(f"Auto-save failed: {e}")  # Don't show error dialog for auto-save
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    try:
        app = TextExtractionGUI()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()

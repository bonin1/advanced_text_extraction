"""
Advanced Text Extraction CLI
===========================

Command-line interface for the Advanced Text Extraction Software.
Provides powerful batch processing, progress tracking, and multiple
output formats for automation and scripting.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

# Rich for beautiful CLI output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to basic print
    def rprint(*args, **kwargs):
        print(*args, **kwargs)

# Click for advanced CLI features
try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

from text_extractor import AdvancedTextExtractor, ExtractionResult


class TextExtractionCLI:
    """Command-line interface for text extraction"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.extractor = AdvancedTextExtractor()
        
    def print_banner(self):
        """Print application banner"""
        if RICH_AVAILABLE:
            banner_text = """
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │        🚀 Advanced Text Extraction Software                │
    │                                                             │
    │     Extract text from images, documents, and more!         │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
            """
            panel = Panel(
                banner_text,
                style="bold blue",
                border_style="bright_blue"
            )
            self.console.print(panel)
        else:
            print("=" * 60)
            print("🚀 Advanced Text Extraction Software")
            print("Extract text from images, documents, and more!")
            print("=" * 60)
    
    def extract_single_file(self, file_path: str, output_path: str = None, 
                          format_type: str = 'txt', engine: str = 'auto',
                          verbose: bool = False) -> bool:
        """Extract text from a single file"""
        try:
            if verbose:
                rprint(f"📄 Processing: [bold]{Path(file_path).name}[/bold]")
            
            start_time = time.time()
            result = self.extractor.extract_from_file(file_path)
            processing_time = time.time() - start_time
            
            if verbose:
                rprint(f"✅ Completed in {processing_time:.2f}s")
                rprint(f"   Language: {result.language}")
                rprint(f"   Confidence: {result.confidence:.2f}")
                rprint(f"   Characters: {len(result.text):,}")
                rprint(f"   Words: {len(result.text.split()):,}")
            
            # Output results
            if output_path:
                self._save_single_result(result, output_path, format_type)
                if verbose:
                    rprint(f"💾 Saved to: [bold]{output_path}[/bold]")
            else:
                # Print to stdout
                if format_type == 'json':
                    output_data = {
                        'file_path': result.source_file,
                        'text': result.text,
                        'confidence': result.confidence,
                        'language': result.language,
                        'metadata': result.metadata,
                        'processing_time': result.processing_time,
                        'timestamp': result.timestamp.isoformat()
                    }
                    print(json.dumps(output_data, indent=2, ensure_ascii=False))
                else:
                    print(result.text)
            
            return True
            
        except Exception as e:
            rprint(f"❌ Error processing {file_path}: {str(e)}", style="bold red")
            return False
    
    def extract_batch(self, file_paths: List[str], output_dir: str = None,
                     format_type: str = 'txt', engine: str = 'auto',
                     max_workers: int = 4, verbose: bool = False) -> Dict[str, Any]:
        """Extract text from multiple files"""
        if verbose:
            rprint(f"🚀 Starting batch extraction of {len(file_paths)} files")
            rprint(f"   Engine: {engine}")
            rprint(f"   Workers: {max_workers}")
            rprint(f"   Output format: {format_type}")
        
        # Progress tracking
        if RICH_AVAILABLE and verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                
                task = progress.add_task("Processing files...", total=len(file_paths))
                
                def progress_callback(current, total, current_file):
                    progress.update(task, 
                                  completed=current,
                                  description=f"Processing {Path(current_file).name}")
                
                results = self.extractor.batch_extract(
                    file_paths, 
                    max_workers=max_workers,
                    progress_callback=progress_callback
                )
        else:
            # Simple progress without rich
            results = []
            for i, file_path in enumerate(file_paths, 1):
                if verbose:
                    print(f"Processing {i}/{len(file_paths)}: {Path(file_path).name}")
                
                try:
                    result = self.extractor.extract_from_file(file_path)
                    results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"Error: {e}")
        
        # Save results
        if output_dir:
            self._save_batch_results(results, output_dir, format_type)
        
        # Generate summary
        summary = self._generate_batch_summary(results, file_paths)
        
        if verbose:
            self._print_batch_summary(summary)
        
        return summary
    
    def _save_single_result(self, result: ExtractionResult, output_path: str, format_type: str):
        """Save single extraction result"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            output_data = {
                'file_path': result.source_file,
                'text': result.text,
                'confidence': result.confidence,
                'language': result.language,
                'extraction_method': result.extraction_method,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'timestamp': result.timestamp.isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
        else:  # txt format
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Source: {result.source_file}\\n")
                f.write(f"Language: {result.language}\\n")
                f.write(f"Confidence: {result.confidence:.2f}\\n")
                f.write(f"Method: {result.extraction_method}\\n")
                f.write(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write("-" * 50 + "\\n")
                f.write(result.text)
    
    def _save_batch_results(self, results: List[ExtractionResult], output_dir: str, format_type: str):
        """Save batch extraction results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual files
        for i, result in enumerate(results):
            source_name = Path(result.source_file).stem
            if format_type == 'json':
                output_file = output_dir / f"{source_name}_extracted.json"
            else:
                output_file = output_dir / f"{source_name}_extracted.txt"
            
            self._save_single_result(result, output_file, format_type)
        
        # Save combined results
        if format_type == 'json':
            combined_file = output_dir / "combined_results.json"
            self.extractor.export_results(results, str(combined_file), 'json')
        else:
            combined_file = output_dir / "combined_results.txt"
            self.extractor.export_results(results, str(combined_file), 'txt')
    
    def _generate_batch_summary(self, results: List[ExtractionResult], 
                               file_paths: List[str]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing"""
        total_files = len(file_paths)
        successful_files = len(results)
        failed_files = total_files - successful_files
        
        if results:
            total_chars = sum(len(r.text) for r in results)
            total_words = sum(len(r.text.split()) for r in results)
            avg_confidence = sum(r.confidence for r in results) / len(results)
            total_time = sum(r.processing_time for r in results)
            
            # Language distribution
            languages = {}
            for result in results:
                lang = result.language
                languages[lang] = languages.get(lang, 0) + 1
            
            # Method distribution
            methods = {}
            for result in results:
                method = result.extraction_method
                methods[method] = methods.get(method, 0) + 1
        else:
            total_chars = total_words = avg_confidence = total_time = 0
            languages = methods = {}
        
        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': (successful_files / total_files * 100) if total_files > 0 else 0,
            'total_characters': total_chars,
            'total_words': total_words,
            'average_confidence': avg_confidence,
            'total_processing_time': total_time,
            'languages': languages,
            'methods': methods
        }
    
    def _print_batch_summary(self, summary: Dict[str, Any]):
        """Print batch processing summary"""
        if RICH_AVAILABLE:
            # Create summary table
            table = Table(title="📊 Extraction Summary", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Total Files", str(summary['total_files']))
            table.add_row("Successful", str(summary['successful_files']))
            table.add_row("Failed", str(summary['failed_files']))
            table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
            table.add_row("Total Characters", f"{summary['total_characters']:,}")
            table.add_row("Total Words", f"{summary['total_words']:,}")
            table.add_row("Avg Confidence", f"{summary['average_confidence']:.2f}")
            table.add_row("Processing Time", f"{summary['total_processing_time']:.2f}s")
            
            self.console.print(table)
            
            # Language distribution
            if summary['languages']:
                lang_table = Table(title="🌍 Language Distribution")
                lang_table.add_column("Language", style="cyan")
                lang_table.add_column("Files", style="magenta")
                lang_table.add_column("Percentage", style="yellow")
                
                for lang, count in sorted(summary['languages'].items(), 
                                        key=lambda x: x[1], reverse=True):
                    percentage = (count / summary['successful_files']) * 100
                    lang_table.add_row(lang, str(count), f"{percentage:.1f}%")
                
                self.console.print(lang_table)
            
        else:
            # Simple text output
            print("\\n" + "=" * 50)
            print("📊 EXTRACTION SUMMARY")
            print("=" * 50)
            print(f"Total Files: {summary['total_files']}")
            print(f"Successful: {summary['successful_files']}")
            print(f"Failed: {summary['failed_files']}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Total Characters: {summary['total_characters']:,}")
            print(f"Total Words: {summary['total_words']:,}")
            print(f"Average Confidence: {summary['average_confidence']:.2f}")
            print(f"Processing Time: {summary['total_processing_time']:.2f}s")
            
            if summary['languages']:
                print("\\nLanguage Distribution:")
                for lang, count in sorted(summary['languages'].items(), 
                                        key=lambda x: x[1], reverse=True):
                    percentage = (count / summary['successful_files']) * 100
                    print(f"  {lang}: {count} files ({percentage:.1f}%)")
    
    def list_supported_formats(self):
        """List supported file formats"""
        if RICH_AVAILABLE:
            table = Table(title="📁 Supported File Formats")
            table.add_column("Category", style="cyan")
            table.add_column("Extensions", style="magenta")
            table.add_column("Description", style="yellow")
            
            table.add_row(
                "Images",
                ".jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp",
                "OCR text extraction from images"
            )
            table.add_row(
                "Documents",
                ".pdf, .docx, .xlsx, .pptx",
                "Text extraction from office documents"
            )
            table.add_row(
                "Text",
                ".txt, .md, .rst",
                "Plain text files"
            )
            
            self.console.print(table)
        else:
            print("\\nSupported File Formats:")
            print("-" * 30)
            print("Images: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp")
            print("Documents: .pdf, .docx, .xlsx, .pptx")
            print("Text: .txt, .md, .rst")
    
    def show_history(self, limit: int = 10):
        """Show extraction history"""
        history = self.extractor.get_extraction_history(limit)
        
        if not history:
            rprint("📜 No extraction history found.", style="yellow")
            return
        
        if RICH_AVAILABLE:
            table = Table(title=f"📜 Recent Extractions (Last {len(history)})")
            table.add_column("File", style="cyan", max_width=30)
            table.add_column("Method", style="magenta")
            table.add_column("Language", style="yellow")
            table.add_column("Confidence", style="green")
            table.add_column("Date", style="blue")
            
            for entry in history:
                filename = Path(entry['file_path']).name
                if len(filename) > 27:
                    filename = filename[:24] + "..."
                
                table.add_row(
                    filename,
                    entry['extraction_method'],
                    entry['language'],
                    f"{entry['confidence']:.2f}",
                    entry['timestamp'][:16]  # Date and time only
                )
            
            self.console.print(table)
        else:
            print(f"\\nRecent Extractions (Last {len(history)}):")
            print("-" * 80)
            for entry in history:
                filename = Path(entry['file_path']).name
                print(f"{filename:<30} {entry['extraction_method']:<10} "
                      f"{entry['language']:<8} {entry['confidence']:.2f} "
                      f"{entry['timestamp'][:16]}")


def find_files_by_pattern(patterns: List[str], recursive: bool = False) -> List[str]:
    """Find files matching glob patterns"""
    files = []
    
    for pattern in patterns:
        pattern_path = Path(pattern)
        
        if pattern_path.is_file():
            files.append(str(pattern_path.resolve()))
        elif pattern_path.is_dir():
            # Directory - find all supported files
            supported_extensions = {
                '.pdf', '.docx', '.xlsx', '.pptx', '.txt',
                '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
            }
            
            if recursive:
                search_pattern = "**/*"
            else:
                search_pattern = "*"
            
            for file_path in pattern_path.glob(search_pattern):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    files.append(str(file_path.resolve()))
        else:
            # Glob pattern
            from glob import glob
            matched_files = glob(pattern, recursive=recursive)
            files.extend([str(Path(f).resolve()) for f in matched_files if Path(f).is_file()])
    
    return sorted(list(set(files)))  # Remove duplicates and sort


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Text Extraction Software - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract text from a single file
  python cli.py extract document.pdf
  
  # Extract from multiple files with output directory
  python cli.py extract *.pdf -o results/ -f json
  
  # Batch processing with custom settings
  python cli.py extract folder/ -r -w 8 -e easyocr -v
  
  # Show extraction history
  python cli.py history -n 20
  
  # List supported formats
  python cli.py formats
        """
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract text from files')
    extract_parser.add_argument(
        'files',
        nargs='+',
        help='Files, directories, or glob patterns to process'
    )
    extract_parser.add_argument(
        '-o', '--output',
        help='Output file or directory'
    )
    extract_parser.add_argument(
        '-f', '--format',
        choices=['txt', 'json'],
        default='txt',
        help='Output format (default: txt)'
    )
    extract_parser.add_argument(
        '-e', '--engine',
        choices=['auto', 'tesseract', 'easyocr', 'paddleocr'],
        default='auto',
        help='OCR engine to use (default: auto)'
    )
    extract_parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    extract_parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process directories recursively'
    )
    extract_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show extraction history')
    history_parser.add_argument(
        '-n', '--number',
        type=int,
        default=10,
        help='Number of recent entries to show (default: 10)'
    )
    
    # Formats command
    subparsers.add_parser('formats', help='List supported file formats')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize CLI
    cli = TextExtractionCLI()
    
    if not args.command:
        cli.print_banner()
        parser.print_help()
        return
    
    # Handle commands
    if args.command == 'extract':
        cli.print_banner()
        
        # Find files
        files = find_files_by_pattern(args.files, args.recursive)
        
        if not files:
            rprint("❌ No files found matching the specified patterns.", style="bold red")
            return
        
        if args.verbose:
            rprint(f"📁 Found {len(files)} files to process")
        
        # Process files
        if len(files) == 1 and not args.output:
            # Single file to stdout
            cli.extract_single_file(
                files[0],
                output_path=args.output,
                format_type=args.format,
                engine=args.engine,
                verbose=args.verbose
            )
        else:
            # Batch processing
            summary = cli.extract_batch(
                files,
                output_dir=args.output,
                format_type=args.format,
                engine=args.engine,
                max_workers=args.workers,
                verbose=args.verbose
            )
            
            if summary['failed_files'] > 0:
                sys.exit(1)  # Exit with error code if some files failed
    
    elif args.command == 'history':
        cli.show_history(args.number)
    
    elif args.command == 'formats':
        cli.list_supported_formats()


if __name__ == "__main__":
    main()

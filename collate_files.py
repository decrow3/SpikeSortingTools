#!/usr/bin/env python3
"""
Collate all Python and Markdown files from the repository into a single text file.
"""

from pathlib import Path
import sys

def collate_files(repo_path, output_file):
    """Collate all .py and .md files into a single text file."""
    repo_path = Path(repo_path)
    output_path = Path(output_file)
    
    # Find all Python and Markdown files
    py_files = sorted(repo_path.rglob('*.py'))
    md_files = sorted(repo_path.rglob('*.md'))
    all_files = py_files + md_files
    
    # Filter out __pycache__ and common unwanted directories
    excluded_dirs = {'__pycache__', '.git', '.pytest_cache', 'node_modules', '.venv', 'venv'}
    all_files = [
        f for f in all_files 
        if not any(excluded in f.parts for excluded in excluded_dirs)
    ]
    
    print(f"Found {len(all_files)} files to collate")
    
    with open(output_path, 'w', encoding='utf-8') as outf:
        outf.write("=" * 80 + "\n")
        outf.write("REPOSITORY CODE AND DOCUMENTATION COLLATION\n")
        outf.write("=" * 80 + "\n\n")
        
        for i, filepath in enumerate(all_files, 1):
            relative_path = filepath.relative_to(repo_path)
            outf.write("\n" + "=" * 80 + "\n")
            outf.write(f"FILE {i}/{len(all_files)}: {relative_path}\n")
            outf.write("=" * 80 + "\n\n")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as inf:
                    content = inf.read()
                    outf.write(content)
            except Exception as e:
                outf.write(f"[ERROR reading file: {e}]\n")
            
            outf.write("\n")
    
    print(f"Collation complete! Output written to: {output_path}")
    print(f"Total files processed: {len(all_files)}")

if __name__ == "__main__":
    # Default to current directory if no argument provided
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    output_file = sys.argv[2] if len(sys.argv) > 2 else "repo_collation.txt"
    
    collate_files(repo_path, output_file)
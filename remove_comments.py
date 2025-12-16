import os
import re
from pathlib import Path

def remove_comments_and_docstrings(code):
    lines = code.split('\n')
    result = []
    in_docstring = False
    docstring_char = None
    skip_next_empty = False
    
    for line in lines:
        stripped = line.strip()
        
        if in_docstring:
            if docstring_char in stripped:
                if stripped.endswith(docstring_char):
                    in_docstring = False
                    docstring_char = None
                    skip_next_empty = True
            continue
        
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                skip_next_empty = True
                continue
            in_docstring = True
            docstring_char = '"""' if stripped.startswith('"""') else "'''"
            continue
        
        if stripped.startswith('#'):
            continue
        
        if '#' in line and not ('"' in line or "'" in line):
            code_part = line.split('#')[0].rstrip()
            if code_part:
                result.append(code_part)
            continue
        
        if skip_next_empty and not stripped:
            skip_next_empty = False
            continue
        
        result.append(line)
    
    return '\n'.join(result)

def process_file(filepath):
    print(f"Processing: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned = remove_comments_and_docstrings(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    print(f"  ✓ Cleaned")

src_dir = Path('src')
for py_file in src_dir.glob('*.py'):
    if py_file.name != '__pycache__':
        process_file(py_file)

for py_file in [Path('demo.py'), Path('test_system.py')]:
    if py_file.exists():
        process_file(py_file)

print("\n✓ All comments and docstrings removed!")

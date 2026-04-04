import os
import re
from pathlib import Path

def comprehensive_html_check():
    """Comprehensive check for HTML rendering issues"""
    app_dir = Path("app")
    issues = []
    
    for py_file in app_dir.rglob("*.py"):
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for any HTML-related calls
            if any(pattern in line for pattern in ['st.write(', 'st.text(', 'print(', 'return f"""<']):
                if '<' in line and '>' in line:
                    # Check if it has proper escaping
                    if 'unsafe_allow_html' not in '\n'.join(lines[max(0,i-5):min(len(lines),i+5)]):
                        issues.append({
                            'file': py_file,
                            'line': i,
                            'code': line.strip()[:80],
                            'concerns': ['Possible missing unsafe_allow_html flag']
                        })
            
            # Also check for any place where f-strings might be generating HTML
            if 'f"""<' in line or "f'''<" in line:
                # Look ahead to see if it's properly wrapped
                block = '\n'.join(lines[max(0,i-2):min(len(lines),i+10)])
                if 'unsafe_allow_html=True' not in block:
                    issues.append({
                        'file': py_file,
                        'line': i,
                        'code': line.strip()[:80],
                        'concerns': ['F-string HTML without unsafe_allow_html']
                    })

    return issues

if __name__ == "__main__":
    issues = comprehensive_html_check()
    
    if issues:
        print(f"⚠️  Potential HTML rendering issues found:\n")
        for issue in issues:
            print(f"📁 {issue['file']}:{issue['line']}")
            print(f"   Code: {issue['code']}")
            print(f"   Concerns: {', '.join(issue['concerns'])}")
            print()
    else:
        print("✅ HTML rendering appears proper throughout the codebase!")

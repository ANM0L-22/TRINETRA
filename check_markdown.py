import os
import re
from pathlib import Path

def find_markdown_issues():
    """Find all st.markdown() calls with HTML but missing unsafe_allow_html=True"""
    app_dir = Path("app")
    issues = []
    
    for py_file in app_dir.rglob("*.py"):
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Find st.markdown calls
        for i, line in enumerate(lines, 1):
            if 'st.markdown(' in line:
                # Look ahead to see if this is a multi-line call
                start_line = i - 1
                markdown_block = line
                j = i
                paren_count = line.count('(') - line.count(')')
                
                # Collect multi-line markdown calls
                while paren_count > 0 and j < len(lines):
                    markdown_block += '\n' + lines[j]
                    paren_count += lines[j].count('(') - lines[j].count(')')
                    j += 1
                
                # Check if it contains HTML tags and check for unsafe_allow_html
                has_html = bool(re.search(r'<[a-z]', markdown_block, re.IGNORECASE))
                has_flag = 'unsafe_allow_html=True' in markdown_block or 'unsafe_allow_html = True' in markdown_block
                
                if has_html and not has_flag:
                    # Show snippet
                    snippet = markdown_block[:100].replace('\n', ' ')
                    issues.append({
                        'file': py_file,
                        'line': i,
                        'snippet': snippet,
                        'full': markdown_block
                    })
    
    return issues

if __name__ == "__main__":
    issues = find_markdown_issues()
    
    if issues:
        print(f"🔍 Found {len(issues)} st.markdown() calls with HTML but missing unsafe_allow_html=True:\n")
        for issue in issues:
            print(f"📁 {issue['file']}:{issue['line']}")
            print(f"   {issue['snippet'][:80]}...")
            print()
    else:
        print("✅ All st.markdown() calls with HTML have unsafe_allow_html=True!")

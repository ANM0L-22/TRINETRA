import re
from pathlib import Path

def find_true_markdown_issues():
    """Find ONLY st.markdown() calls that have HTML tags but lack unsafe_allow_html=True"""
    app_dir = Path("app")
    issues = []
    
    for py_file in app_dir.rglob("*.py"):
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Find all st.markdown() calls
        pattern = r'st\.markdown\(((?:[^()]*|\([^()]*\))*)\)'
        for match in re.finditer(pattern, content, re.DOTALL):
            full_call = match.group(0)
            call_content = match.group(1)
            
            # Check if contains HTML tags and lacks the flag
            has_html_tags = bool(re.search(r'<\s*(?:div|span|style|hr|br|img|button|input|select|textarea|form|table|p|h\d)[^>]*>', call_content, re.IGNORECASE))
            has_flag = bool(re.search(r'unsafe_allow_html\s*=\s*True', call_content))
            
            if has_html_tags and not has_flag:
                # Get line number
                line_num = content[:match.start()].count('\n') + 1
                issues.append({
                    'file': py_file,
                    'line': line_num,
                    'snippet': full_call[:150],
                    'has_html': has_html_tags
                })
    
    return issues

if __name__ == "__main__":
    issues = find_true_markdown_issues()
    
    if issues:
        print(f"⚠️  Found {len(issues)} st.markdown() calls needing unsafe_allow_html=True:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue['file']}:{issue['line']}")
            preview = issue['snippet'].replace('\n', ' ')[:100]
            print(f"   {preview}...")
            print()
        
        # Generate fix script
        print("\n" + "="*60)
        print("📝 LOCATIONS TO FIX:")
        for issue in issues:
            print(f"  - {issue['file']}:{issue['line']}")
    else:
        print("✅ All st.markdown() calls with HTML have unsafe_allow_html=True!")

import re

with open('app/pages/dashboard.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Replace use_container_width=True with width='stretch'
content = re.sub(r'use_container_width=True', "width='stretch'", content)
# Replace use_container_width=False with width='content'
content = re.sub(r'use_container_width=False', "width='content'", content)

with open('app/pages/dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully replaced all use_container_width with width parameter')

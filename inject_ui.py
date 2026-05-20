"""Inject new UI HTML into integrated_test_app.py using byte-level operations."""
import re

# Read as bytes
with open('integrated_test_app.py', 'rb') as f:
    data = f.read()

# Read the new HTML template
with open('new_ui.html', 'r', encoding='utf-8') as f:
    new_html = f.read()

# Find INDEX_HTML in bytes
# Pattern: INDEX_HTML = followed by 3 fancy quotes (U+201C or U+201D)
idx_start = data.find(b'INDEX_HTML')
if idx_start == -1:
    print("ERROR: INDEX_HTML not found")
    exit(1)

# Find the start of the HTML string (after the triple quotes)
# Triple quotes are ~9 bytes: space = \xe2\x80\x9c\xe2\x80\x9d\xe2\x80\x9d\n
# Find the first \n after INDEX_HTML
newline_after = data.find(b'\n', idx_start)
html_start = newline_after + 1  # First char of HTML: '<'

# Find the closing triple quotes: find """ before @app.get
# Look for pattern: \n"""\n\n@app.get
# In fancy quotes: \n\xe2\x80\x9c\xe2\x80\x9d\xe2\x80\x9d\n\n@app.get
closing_patterns = [
    b'\n\xe2\x80\x9c\xe2\x80\x9d\xe2\x80\x9d\n\n@app.get',
    b'\n\xe2\x80\x9c\xe2\x80\x9d\xe2\x80\x9d\n\n\n@app.get',
    b'\n"""\n\n@app.get',
    b'\n"""\n\n\n@app.get',
]
html_end = -1
for pat in closing_patterns:
    pos = data.find(pat, html_start + 5000)
    if pos != -1:
        html_end = pos + 1  # Position right after first \n of pattern
        break

if html_end == -1:
    # Fallback: find triple fancy quotes near end
    fancy_triple = b'\xe2\x80\x9c\xe2\x80\x9d\xe2\x80\x9d'
    pos = data.rfind(fancy_triple, html_start + 5000)
    if pos != -1:
        html_end = pos
    else:
        # Try ASCII triple quotes
        pos = data.rfind(b'"""', html_start + 5000)
        if pos != -1:
            html_end = pos
        else:
            print("ERROR: Could not find closing quotes")
            exit(1)

print(f"HTML start: {html_start}, HTML end: {html_end}")
print(f"Old HTML size: {html_end - html_start} bytes")

# Extract the JavaScript from the old HTML
old_html = data[html_start:html_end].decode('utf-8', errors='replace')
script_match = re.search(r'<script>(.*?)</script>', old_html, re.DOTALL)
if script_match:
    old_js = '<script>' + script_match.group(1) + '</script>'
    print(f"Extracted JS: {len(old_js)} chars")
else:
    print("WARNING: Could not extract JavaScript!")
    old_js = "<script>\n</script>"

# Insert JS into new HTML
body_pos = new_html.find('</body>')
if body_pos != -1:
    final_html = new_html[:body_pos] + '\n' + old_js + '\n' + new_html[body_pos:]
else:
    final_html = new_html + '\n' + old_js

# Build new content
prefix = data[:html_start]
suffix = data[html_end:]

new_html_bytes = final_html.encode('utf-8')
new_data = prefix + new_html_bytes + suffix

# Write back
with open('integrated_test_app.py', 'wb') as f:
    f.write(new_data)

print(f"Done! New file size: {len(new_data)} bytes")
print(f"New HTML size: {len(new_html_bytes)} bytes")

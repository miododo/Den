"""Inject new UI into integrated_test_app.py - no fancy quotes this time."""
import re

with open('integrated_test_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

with open('new_ui.html', 'r', encoding='utf-8') as f:
    new_html = f.read()

# Find boundaries using ASCII quotes
idx_start = content.find('INDEX_HTML = """')
assert idx_start != -1, "INDEX_HTML start not found"

# Find the closing """ before @app.get
# Need to find the one that's followed by @app.get
html_content_start = content.find('\n', idx_start) + 1
# Find closing: look for pattern \n"""\n\n@app.get
closing = content.find('\n"""\n\n@app.get', html_content_start)
if closing == -1:
    closing = content.find('\n"""\n\n\n@app.get', html_content_start)
assert closing != -1, "Closing not found"

# Extract JS from original
old_html = content[html_content_start:closing]
script_match = re.search(r'<script>(.*?)</script>', old_html, re.DOTALL)
old_js = '<script>' + script_match.group(1) + '</script>' if script_match else '<script></script>'
print(f"Extracted JS: {len(old_js)} chars")

# Read additional new JS
with open('new_js.js', 'r', encoding='utf-8') as f:
    new_js = f.read()
# Inject new JS before closing </script>
old_js = old_js.replace('</script>', new_js + '\n</script>')
print(f"Combined JS: {len(old_js)} chars")

# Combine: insert JS before </body> of new HTML
body_close = new_html.find('</body>')
if body_close != -1:
    final_html = new_html[:body_close] + '\n' + old_js + '\n' + new_html[body_close:]
else:
    final_html = new_html + '\n' + old_js

# Now replace in content
new_content = content[:html_content_start] + final_html + '\n"""' + content[closing + len('\n"""'):]

# Fix: make sure @app.get is still there
if '\n@app.get("/"' not in new_content:
    # We removed too much, fix it
    proper_end = content.find('\n@app.get("/"', closing)
    new_content = content[:html_content_start] + final_html + '\n"""' + content[proper_end:]
    # But we need a newline before @app.get
    new_content = new_content.replace('"""@app.get', '"""\n\n\n@app.get')

# Verify no fancy quotes in the file
fancy_left = '“'
fancy_right = '”'
if fancy_left in new_content or fancy_right in new_content:
    print("WARNING: fancy quotes still present, removing...")
    # Only fix the triple-quote delimiters, not Chinese text
    # Find INDEX_HTML line
    idx = new_content.find('INDEX_HTML')
    line_end = new_content.find('\n', idx)
    idx_line = new_content[idx:line_end]
    fixed_line = idx_line.replace('“', '"').replace('”', '"')
    new_content = new_content[:idx] + fixed_line + new_content[line_end:]
    # Find closing triple quotes
    closing_idx = new_content.rfind('“””', 0, new_content.rfind('@app.get'))
    if closing_idx != -1:
        new_content = new_content[:closing_idx] + '"""' + new_content[closing_idx+len('“””'):]

# Write back
with open('integrated_test_app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

# Verify syntax
import py_compile
try:
    py_compile.compile('integrated_test_app.py', doraise=True)
    print("Syntax check: PASS")
except py_compile.PyCompileError as e:
    print(f"Syntax error: {e}")
    exit(1)

print(f"Done! New file size: {len(new_content)} chars")
print(f"New HTML size: {len(final_html)} chars")

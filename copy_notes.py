import os
import shutil
import glob

os.makedirs('memory/stock-trading', exist_ok=True)
download_dir = os.path.expanduser('~/Downloads')

# Copy strategy and glossary files
files_to_copy = glob.glob(os.path.join(download_dir, 'strategy*.md')) + glob.glob(os.path.join(download_dir, 'glossary.md'))

for file_path in files_to_copy:
    dest_path = os.path.join('memory/stock-trading', os.path.basename(file_path))
    shutil.copy(file_path, dest_path)
    print(f"Successfully copied {os.path.basename(file_path)} to memory/stock-trading/")

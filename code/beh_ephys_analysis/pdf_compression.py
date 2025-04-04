import subprocess
import os
from utils.beh_functions import session_dirs
def compress_pdf_ghostscript(input_path, output_path, quality="screen"):
    gs_command = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS=/{quality}",  # Options: screen, ebook, printer, prepress
        "-dNOPAUSE", "-dBATCH", "-sOutputFile=" + output_path, input_path
    ]
    subprocess.run(gs_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Compressed PDF saved as: {output_path}")

# Usage
session = 'behavior_751766_2025-02-13_11-31-21'
session_dir = session_dirs(session)
data_type = 'raw'
file = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging.pdf')
file_compressed = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging_compressed.pdf')
print(file)
print(file_compressed)
compress_pdf_ghostscript(file, file_compressed)

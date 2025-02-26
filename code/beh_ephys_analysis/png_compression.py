import os
from pdf2image import convert_from_path
from PIL import Image
import os
from utils.beh_functions import session_dirs

# Define the directory containing PDFs
session = 'behavior_751766_2025-02-13_11-31-21'
data_type = 'raw'
def opto_pdf_compression(session, data_type):
    pdf_dir = session_dirs(session)[f'opto_dir_fig_{data_type}']
    output_pdf = os.path.join(session_dirs(session)[f'opto_dir_{data_type}'], f'{session}_opto_tagging_png.pdf')

    # Ensure the output directory exists
    output_images_dir = os.path.join(pdf_dir, "converted_images")
    os.makedirs(output_images_dir, exist_ok=True)

    png_images = []

    # Convert each PDF to PNG
    for file in sorted(os.listdir(pdf_dir)):  # Sort to maintain order
        print(f'Processing file: {file}')
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file)
            images = convert_from_path(pdf_path, dpi=300)  # High-quality conversion
            
            for i, img in enumerate(images):
                img_path = os.path.join(output_images_dir, f"{file[:-4]}_page_{i+1}.png")
                img.save(img_path, "PNG")
                png_images.append(img_path)  # Store paths for merging later

    # Combine all PNGs into a single PDF
    if png_images:
        image_objs = [Image.open(img).convert("RGB") for img in png_images]
        image_objs[0].save(os.path.join(pdf_dir, output_pdf), save_all=True, append_images=image_objs[1:])
        print(f"Combined PDF saved as: {os.path.join(pdf_dir, output_pdf)}")
    else:
        print("No PDFs found in the directory!")
    
    

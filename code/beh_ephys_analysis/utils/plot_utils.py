import numpy as np
from scipy import stats
import statsmodels.api as sm
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os
from scipy.stats import norm
from datetime import datetime
import ast
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from natsort import natsorted
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import shutil
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def shiftedColorMap(cmap, min_val, max_val, name):
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    midpoint = 1.0 - max_val/(max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = LinearSegmentedColormap(name, cdict)

    # colormaps.register(cmap=newcmap, force=True)
    return newcmap

# Interpolate all waveforms
def interpolate_waveform(waveform):
    # Create an array of the desired shape
    interpolated_waveform = np.zeros(np.shape(waveform))
    for col_ind in range(waveform.shape[1]):
        nan_ind = np.isnan(waveform[:, col_ind])
        if np.sum(nan_ind) > 0:
            # Interpolate the waveform for each column
            temp = np.interp(np.arange(waveform.shape[0]), np.arange(waveform.shape[0])[~nan_ind], waveform[~nan_ind, col_ind])
            interpolated_waveform[:, col_ind] = temp
        else:
            interpolated_waveform[:, col_ind] = waveform[:, col_ind]
    return interpolated_waveform
    
def template_reorder(template, right_left, all_channels_int, sample_to_keep = [-30, 60], y_neighbors_to_keep = 3, orginal_loc = False, peak_ind = None):
    if peak_ind is None:
        peak_ind = np.argmin(np.min(template, axis=0))
    peak_channel = all_channels_int[peak_ind]
    peak_sample = np.argmin(template[:, peak_ind])  
    peak_group = np.arange(peak_channel - 2*y_neighbors_to_keep, peak_channel + 2*y_neighbors_to_keep + 1, 2)
    if right_left[peak_ind]: # peak is on the right
        sub_peak_channel = peak_channel - 1 
    else: 
        sub_peak_channel = peak_channel + 1 
    sub_group = np.arange(sub_peak_channel - 2*y_neighbors_to_keep, sub_peak_channel + 2*y_neighbors_to_keep + 1, 2)

    # get the reordered template: major column on left, minor column on right
    reordered_template = np.full((2*y_neighbors_to_keep + 1, 2*(sample_to_keep[1] - sample_to_keep[0])), np.nan)
    if peak_sample+sample_to_keep[1] > template.shape[0] or peak_sample+sample_to_keep[0] < 0:
        peak_sample = np.round(1/3 * template.shape[0]).astype(int)
    for channel_int, channel_curr in enumerate(peak_group):
        if channel_curr in all_channels_int:
            reordered_template[channel_int, :(sample_to_keep[1] - sample_to_keep[0])] = template[(peak_sample+sample_to_keep[0]):(peak_sample+sample_to_keep[1]), np.argwhere(all_channels_int == channel_curr)[0][0]].T

    for channel_int, channel_curr in enumerate(sub_group):
        if channel_curr in all_channels_int:
            reordered_template[channel_int, (sample_to_keep[1] - sample_to_keep[0]):2*(sample_to_keep[1] - sample_to_keep[0])] = template[(peak_sample+sample_to_keep[0]):(peak_sample+sample_to_keep[1]), np.argwhere(all_channels_int == channel_curr)[0][0]].T

    # whether switch back to original location
    if orginal_loc:
        if right_left[peak_ind]:
            reordered_template = reordered_template[:, list(range((sample_to_keep[1] - sample_to_keep[0]), 2*(sample_to_keep[1] - sample_to_keep[0]))) + list(range((sample_to_keep[1] - sample_to_keep[0])))]
    return reordered_template

def plot_raster_bar(raster_df, ax):
    events = raster_df['event_index'].unique()
    # Plot the raster plot
    for event_ind in events:
        spike_aligned_curr = raster_df.query('event_index == @event_ind')['time'].values
        ax.vlines(spike_aligned_curr, event_ind + 0.5, event_ind + 1.5, color='black', linewidth=1)  # Vertical lines for each event

def merge_pdfs(input_dir, output_filename='merged.pdf'):
    merger = PdfMerger()
    files = os.listdir(input_dir)
    files = natsorted(files)
    # Iterate through all PDF files in the input directory
    for i, filename in enumerate(files):
        if filename.endswith('.pdf'):
            if i%50 == 0:
                print(f'Merging file {i} out of {len(os.listdir(input_dir))}')
            filepath = os.path.join(input_dir, filename)
            merger.append(filepath)

    # Write the merged PDF to the output file
    with open(output_filename, 'wb') as output_file:
        merger.write(output_file)

    print(f"PDF files in '{input_dir}' merged into '{output_filename}' successfully.")

def combine_pdf_big(pdf_dir, output_pdf, add_title= 'default'):
    png_images = []
    
    # Ensure the output directory exists
    output_images_dir = os.path.join(pdf_dir, "converted_images")
    os.makedirs(output_images_dir, exist_ok=True)

    # Load a default font (adjust size as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # Ensure you have Arial installed
    except IOError:
        font = ImageFont.load_default()

    # Convert each PDF to PNG
    files = os.listdir(pdf_dir)
    files = natsorted(files)  # Ensure order
    print(f'Processing {len(files)} files in {pdf_dir}')

    for i, file in enumerate(files):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file)
            images = convert_from_path(pdf_path, dpi=300)  # High-quality conversion

            for i, img in enumerate(images):
                # Add title text on each page
                draw = ImageDraw.Draw(img)
                if add_title == 'default':
                    pattern = r'\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
                    # Find all matches
                    matches = re.search(pattern, file)
                    if matches:
                        title = matches.group(0)
                    else:
                        title = str(i)
                elif isinstance(add_title, list) and len(add_title) == len(files):
                    title = add_title[i]
                elif isinstance(add_title, str):
                    matches = re.search(pattern, file)
                    if matches:
                        title = matches.group(0) + f'_{add_title}'
                    else:
                        title = add_title
                else: 
                    title = str(i)
                text_position = (50, 50)  # Adjust position as needed
                text_color = "black"

                # Add background for better visibility
                text_size = draw.textbbox((0, 0), title, font=font)  # Bounding box of text
                bg_x1, bg_y1, bg_x2, bg_y2 = text_size
                draw.rectangle([text_position, (text_position[0] + bg_x2, text_position[1] + bg_y2)], fill="white")
                
                draw.text(text_position, title, fill=text_color, font=font)

                # Save the modified image
                img_path = os.path.join(output_images_dir, f"{file[:-4]}_page_{i+1}.png")
                img.save(img_path, "PNG")
                png_images.append(img_path)

    # Combine all PNGs into a single PDF
    if png_images:
        image_objs = [Image.open(img).convert("RGB") for img in png_images]
        image_objs[0].save(os.path.join(pdf_dir, output_pdf), save_all=True, append_images=image_objs[1:])
        print(f"Combined PDF saved as: {os.path.join(pdf_dir, output_pdf)}")
    else:
        print("No PDFs found in the directory!")

    # Cleanup temporary images
    shutil.rmtree(output_images_dir)


def get_gradient_colors(max_p_values, floor=0, ceiling=1, cmap_name='Reds'):
    norm = mcolors.Normalize(vmin=floor, vmax=ceiling)
    cmap = cm.get_cmap(cmap_name)  # Use cm.get_cmap instead of colormaps[cmap_name]
    colors = [cmap(norm(max(0, p))) for p in max_p_values]  
    return colors, norm, cmap

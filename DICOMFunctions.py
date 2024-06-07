'''extract the DICOM metadata from the locally stored files 
(for now) and store them in a local chroma database.'''

import io
from EmbeddingFunctions import *
import pydicom
import numpy
import os
import pydicom
import matplotlib.pyplot as plt
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
import asyncio
import os
import pydicom
import re
import os
import cv2
import pydicom
import argparse
from skimage.filters import threshold_otsu
import numpy as np
from skimage import exposure
import math

config_path = "config.yaml"
config=read_config(config_path)
llm = get_chat_model(config["chat_framework"], config["chat_model"])

template = """Use only the given context to answer the question. 
You can not use any knowledge that isn't present in the context.
If you can't answer the question using only the provided context say 'I don't have enough context to answer the question'". 
Don't make up an answer.
context: {context}
question: {question} according to the provided context?

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

embeddings_framework = config["embeddings_framework"]
embeddings_model = config["embeddings_model"]
chat_framework = config["chat_framework"]
chat_model = config["chat_model"]
temperature = config["temperature"]
delimiter = config["delimiter"]
threshold = config["threshold"]
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]
collection = "dicom_files"

vector_db, collection = initialize_vector_database("vector_db", collection, "vector_db", config["embeddings_framework"], config["embeddings_model"])
videos_path = r"C:\Users\CCIG\joao_mata\videos"

def get_dicom_files(directory):
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files

dicom_element_names = [
    "Series Instance UID",
    "Series Description",
    "Media Storage SOP Class UID",
    "Source Application Entity Title",
    "Image Type",
    "Study Date",
    "Series Date",
    "Study Time",
    "Modality",
    "Manufacturer",
    "Institution Name",
    "Study Description",
    "Manufacturer's Model Name",
    "[Full fidelity]",
    "[Suite id]",
    "Patient's Name",
    "Patient ID",
    "Patient's Birth Date",
    "Patient's Sex",
    "Patient's Age",
    "Patient's Weight",
    "Contrast/Bolus Agent",
    "Scan Options",
    "Slice Thickness",
    "KVP",
    "Spacing Between Slices",
    "Data Collection Diameter",
    "Software Versions",
    "Contrast/Bolus Route",
    "Exposure Time",
    "X-Ray Tube Current",
    "Exposure",
    "Filter Type",
    "Convolution Kernel",
    "Patient Position",
    "Laterality",
    "Image Comments",
    "Samples per Pixel",
    "Photometric Interpretation",
    "Rows",
    "Columns",
    "Pixel Spacing"
]

def extract_metadata(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    metadata = []
    for element in ds:
        if element.name in dicom_element_names:
           metadata.append(f"{element.name} is {element.value}")
    return " and ".join(metadata)


import os

def get_dicom_files(directory):
    dicom_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    return dicom_files

def add_dicom_to_db(directory, collection, embeddings_model):
    metadata_list = []
    dicom_files = get_dicom_files(directory)
    
    for file in tqdm(dicom_files, desc="Processing DICOM files"):
        data = extract_metadata(file)
        response = ollama.embeddings(model=embeddings_model, prompt=data)
        embedding = response["embedding"]
        
        
        # Extract the relative path from the top directory
        relative_path = os.path.relpath(file, directory)
        metadata = {"source": relative_path}

        id = relative_path.replace(os.sep, '\\')
        
        collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[data],
            metadatas=[metadata]
        )
        
        metadata_list.append(data)
    
    print("done")
    return metadata_list



def series_into_video(dicom_series, output_dir="videos", frame_rate=1):
    # Create a list to store image file paths
    image_files = []
    series_id = None  # Initialize series_id with a default value
    
    # Read DICOM files, convert to images, and store image paths
    for file in sorted(os.listdir(dicom_series)):
        if file.endswith(".dcm"):
            dicom_path = os.path.join(dicom_series, file)
            ds = pydicom.dcmread(dicom_path)
            img = ds.pixel_array
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            image_filename = os.path.join(dicom_series, f"{os.path.splitext(file)[0]}.jpg")
            cv2.imwrite(image_filename, img_normalized)
            image_files.append(image_filename)
            series_id = ds.SeriesInstanceUID  # Update series_id only if DICOM file is found

    # Check if any DICOM files were found
    if series_id is None:
        print("No DICOM files found in the specified directory.")
        return
    
    # Generate the output video file name with series ID
    output_video = os.path.join(output_dir, f"{series_id.replace('.', '_')}.mp4")
    print("Output video:", output_video)

    # Check if any image files were generated
    if not image_files:
        print("No DICOM files found or no images generated.")
        return
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    
    # Write each image to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is not None:
            video.write(frame)
    
    # Release the video writer object
    video.release()
    
    # Clean up the generated image files
    for image_file in image_files:
        os.remove(image_file)
    
    print(f"Video saved as {output_video}")


def extract_series_description(text):
    # Use regular expression to find the pattern
    match = re.search(r'Series Description is\s*(.*?)\s*and', text)
    if match:
        series_description = match.group(1).replace("_", " ").strip()
        return series_description
    
def extract_study_description(text):
    # Use regular expression to find the pattern
    match = re.search(r'Study Description is\s*(.*?)\s*and', text)
    if match:
        study_description = match.group(1).replace("_", " ").strip()
        return study_description.replace(" ", "")  # Remove all spaces
    else:
        return None
    
def extract_series_id(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    series_id = ds.SeriesInstanceUID
    return series_id
    
def extract_series_id_from_text(text):
    # Use regular expression to find the pattern
    match = re.search(r'Series Instance UID is\s*(.*?)\s*and', text)
    if match:
        series_description = match.group(1).replace("_", " ").strip()
        return series_description.replace(" ", "")  # Remove all spaces
    else:
        return None
    
    
    
    
# PLAYTIME --------------------------------------------------------------
    
import pydicom
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from tqdm import tqdm


def detect_triggers(llm, message):
    options = ["User wants to proccess, segment or detect edges", "user is just chatting and being conversational", "User is asking to see certain image with no transformation", "User is not asking to see or process images, they want information."]
    implemented_methods = ["segmentation", "edge detection"]
    # Define trigger detection prompts
    llm_opinion = llm.invoke(f"Analise the following question: {message} and categorize the message in one of these types: {options}.  You must go over all the list to make sure you choose the most fitting one. Simply repeat the element on the list, no explanation, word by word.")
    if "see" in llm_opinion:
        model_selection = None
        return "show", model_selection
    elif "chatting" in llm_opinion:
        model_selection = None
        return "chat", model_selection
    elif "information" in llm_opinion:
        model_selection = None
        return "questions", model_selection
    elif "proccess" in llm_opinion:
        model_selection = llm.invoke(f"Analise the following question: {message} and decide from this list: {implemented_methods}, which one fits the user's question better.You must simply repeat the element on the list, no explanation, word by word.")            
        return "process", model_selection

#print(detect_triggers(llm, "Can you detect edges on the image?"))
#print(detect_triggers(llm, "Can you use Otsu segmentation on the image?"))
#print(detect_triggers(llm, "Hello what are you doing"))
#print(detect_triggers(llm, "What are the patients ages?"))




def otsu_segmentation(image):
    thresh_val = threshold_otsu(image)
    binary_image = image > thresh_val
    title = "Otsu Thresholding"
    return binary_image, title

def canny_edge_detection(image, low_ths = 100, high_ths=200):
    edges = cv2.Canny(image, low_ths, high_ths)
    title = "Canny Edge Detection"
    return edges, title

def apply_methods(method, series_path):
    print(method)
    image_arrays = []
    series_description = None
    title = None  # Default title
    list_of_slices = os.listdir(series_path)
    for slice in tqdm(list_of_slices, desc="Processing slices"):
        dicom_file = pydicom.dcmread(os.path.join(series_path, slice))
        image = dicom_file.pixel_array.astype(np.uint8)  # Convert pixel array to 8-bit unsigned integer
        
        if series_description is None:
            series_description = dicom_file.SeriesDescription 
        
        if "segmentation" in method:
            binary_image, title = otsu_segmentation(image)
            image_arrays.append(binary_image)

        if "edge" in method:
            edges, title = canny_edge_detection(image)
            image_arrays.append(edges)
            
    print(f"Processed {len(image_arrays)}/{len(list_of_slices)} images")
    return image_arrays, series_description, title


def plot_images(title, images, series_description, filename = r"C:\Users\CCIG\joao_mata\processed_image.png"):
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))  # Number of columns in the grid
    rows = (num_images // cols) + (num_images % cols > 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for i in tqdm(range(num_images), desc="Processing images"):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Image {i}')
        axes[i].axis('off')

    # Hide any empty subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')
    
    plt.suptitle(f'{title}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle
    
    plt.savefig(filename)  # Save the plot as an image file

# TESTING 
filename = r"C:\Users\CCIG\joao_mata\processed_image.png"
series_path = r"C:\Users\CCIG\joao_mata\DICOMS\ProstateX-0001\07-08-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-95738\1.000000-t2localizer-75055"

'''
trigger = detect_triggers(llm, "Can you detect edges on the image?")
print(trigger)
image_arrays, series_description, title = apply_methods(trigger, series_path)
print(series_description)
print(title)
plot_images(title, image_arrays, series_description, filename)
'''



#image_arrays, series_description, title = apply_methods("edges", series_path)
#plot_images(title, image_arrays, series_description, filename)
#print("done")

#for root, dirs, files in os.walk(r”C:\Users\CCIG\joao_mata\DICOMS”):
#    for d in dirs:
#        dicom_series = os.path.join(root, d)
#        series_into_video(dicom_series, videos_path)

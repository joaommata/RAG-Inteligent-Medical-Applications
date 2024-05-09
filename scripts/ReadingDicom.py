import numpy as np
import pydicom
import matplotlib.pyplot as plt

# Reads a DICOM file from the given path and returns the DICOM dataset.
def read_dicom(path):
    ds = pydicom.dcmread(path)
    print(ds)
    return ds

# Reads a DICOM file from the given path and returns a list of DICOM elements.
def get_dicom_elements(path):
    ds = pydicom.dcmread(path)
    elements = dir(ds)
    print(elements)
    return elements

# Reads a DICOM file from the given path and returns the pixel data as a numpy array.
def get_pixel_data(path):    
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    print(pixels)
    return pixels

# Reads a DICOM file from the given path, retrieves the pixel data, and displays the image at the specified index.
def view_dicom(path, index):
    pixels = get_pixel_data(path)
    print(pixels[index])
    plt.imshow(pixels, cmap=plt.cm.bone)
    return 


# TESTING

path = r'C:\Users\CCIG\joao_mata\dicom\0012.dcm'
read_dicom(path)
get_pixel_data(path)
get_dicom_elements(path)
view_dicom(path, 0)


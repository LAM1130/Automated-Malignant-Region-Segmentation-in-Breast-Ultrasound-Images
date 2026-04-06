import cv2
import numpy as np
import os

input_dir = 'malignant'
output_dir = 'output'

def segmentMalignantRegions(img):
    """
    Segments malignant regions in the breast ultrasound image.
    Parameters:
    image (numpy.ndarray): The input ultrasound image.
    Returns:
    mask (numpy.ndarray): A binary mask where malignant regions
    are marked as 1 and other areas as 0.
    """
	
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast, brightness of image
    scaled_img = cv2.convertScaleAbs(img, alpha=2.1, beta=20)

    h, w = scaled_img.shape # Extract the height, width of the image
    lower_third = scaled_img[2*h//3:, :]  # Select the lower third of the image
    brightened_lower_third = cv2.convertScaleAbs(lower_third, alpha=2.5, beta=50)# Further increase the brightness, contrast 
    scaled_img[2*h//3:, :] = brightened_lower_third

    # Otsu's Binarization determines the threshold to separate the foreground and background
    ret,thresh = cv2.threshold(scaled_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Erode Image to further remove noise
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    thresh = cv2.erode(thresh,cross_kernel,iterations = 5)

    # Dilate the image preparing for segmentation
    dilated = cv2.dilate(thresh,cross_kernel,iterations = 5)

    # Find largest contour (lesion)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   #ALL
    largest_cnt = max(contours, key=cv2.contourArea)                                    #Largest

    # Draw the contours
    largest_mask = np.zeros(dilated.shape, np.uint8)
    cv2.drawContours(largest_mask, [largest_cnt], -1, 255, cv2.FILLED)
    malignant_segment = cv2.bitwise_and(dilated, largest_mask)

    # Dilate the image using ellipse kernel (smooth out the edges and connect any fragmented parts)
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation = cv2.dilate(malignant_segment, ellipse_kernel, iterations=5)

    # Create a binary mask where the lesion is marked as 1 and the background as 0
    mask = np.zeros_like(dilation, dtype=np.uint8)
    mask[dilation > 0] = 1

    return mask

    #########################################################################
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  # Check for valid image file extensions
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Segment the image
            segmented_img = segmentMalignantRegions(img)
            
            # Save the segmented image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, segmented_img * 255)
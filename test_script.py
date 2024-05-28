import cv2
import os

# Get the current working directory (where the script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the image file name
image_filename = 'youvegotsoul4.png'  # Change this to the actual image file name

# Create the full path to the image file
image_path = os.path.join(current_dir, image_filename)

# Load the image from file
image = cv2.imread('youvegotsoul4.png')

# Check if the image was successfully loaded
if image is None:
    print("Error: Could not load image.")
else:
    # Display the original image
    cv2.imshow('Original Image', image)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Display the grayscale image
    cv2.imshow('Grayscale Image', gray_image)
    
    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
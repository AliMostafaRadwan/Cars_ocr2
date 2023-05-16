# from PIL import Image
# import os

# def downscale_images(input_folder, output_folder, max_size):
#     """
#     Downscale images in input_folder and save them in output_folder
#     with a maximum size of max_size (in pixels).
#     """
#     # create output_folder if it does not exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # iterate through files in input_folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             # open image
#             image = Image.open(os.path.join(input_folder, filename))
#             # get size of image
#             width, height = image.size
#             # calculate new size with max_size as the maximum dimension
#             if width > height:
#                 new_width = max_size
#                 new_height = int(height * max_size / width)
#             else:
#                 new_height = max_size
#                 new_width = int(width * max_size / height)
#             # downscale image
#             image = image.resize((new_width, new_height), Image.ANTIALIAS)
#             # save downscaled image in output_folder
#             image.save(os.path.join(output_folder, filename))

# # example usage
# input_folder = "pic/CarsLP"
# output_folder = "pic/CarsLP/rescaled"
# max_size = 500
# downscale_images(input_folder, output_folder, max_size)


import cv2
import numpy as np

def enhance_black_color(img_path):
    # Load the image
    img = cv2.imread(img_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute the histogram of the grayscale image
    hist, bins = np.histogram(gray, 256, [0, 256])
    
    # Find the index of the darkest bin with non-zero count
    darkest_bin = np.argmax(hist)
    while hist[darkest_bin] == 0:
        darkest_bin += 1
    
    # Create a lookup table to increase the contrast of the dark regions
    lut = np.zeros((256,), dtype=np.uint8)
    for i in range(256):
        if i < darkest_bin:
            lut[i] = 0
        else:
            lut[i] = 255 * ((i - darkest_bin) / (256 - darkest_bin))
    
    # Apply the lookup table to the grayscale image
    result = cv2.LUT(gray, lut)
    
    # Convert the grayscale result back to BGR color format
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

# example usage

img_path = 'pic/CarsLP/rescaled/6.jpg'
result = enhance_black_color(img_path)
cv2.imshow('result', result)
cv2.waitKey(0)
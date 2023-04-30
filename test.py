
import torch
import numpy as np
import cv2
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """
    
    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_file = '8.jpg'
        print("\n\nDevice Used:",self.device)


    def load_model(self):
        return torch.hub.load('yolov5', model='custom', path='LP_1500.pt', source='local', force_reload=True)



    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
 
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        bboxes = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bboxes.append((x1, y1, x2, y2))
                bgr = (0, 255, 0)
                bgr2 = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr2, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                #print the threshold value
                # cv2.putText(frame, str(row[4]), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr2, 2)
                # st.write(self.class_to_label(labels[i]))
                threash = str(row[4])
                df = pd.DataFrame({'Class': [self.class_to_label(labels[i])], 'Threshold': [threash]})
                st.dataframe(df)
                
        return frame, bboxes
    

    def resize_image(self, image, width, height):
        return cv2.resize(image, (width, height))

    def OCR(self, frame):
        return pytesseract.image_to_string(frame, lang='ara')

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        pic_num = st.number_input("pic", 1, 3000, 1)
        pic_path = 'pic/test/images/{}.jpg'.format(pic_num)


        st.title("Image Manipulation App")
        # read pictures
        
        img_og = cv2.imread(pic_path)
        img = cv2.imread(pic_path)


        # img_resize = cv2.resize(img, (308, 308))
        results = detection.score_frame(img)
        img , bboxes = detection.plot_boxes(results, img)
    
        # Upload image
        # image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        
        if True:
            # Load image
            image = img
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
            # Get user input for new width and height
            new_width = st.number_input("New Width", 10, 3000, 308)
            new_height = st.number_input("New Height", 10, 3000, 308)

            resized_image = self.resize_image(img_og, new_width, new_height)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        

            results2 = detection.score_frame(resized_image)
            img2 , bboxes = detection.plot_boxes(results2, resized_image)
            # Resize image
            
            # Display resized image
            st.subheader("Resized Image")
            st.image(img2, use_column_width=True)



            # Display cropped image
if __name__ == '__main__':
    

    # Create a new object and execute.
    detection = ObjectDetection()
    detection()
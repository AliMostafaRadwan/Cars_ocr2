import torch
import numpy as np
import cv2
import time
import easyocr
import keyboard
from difflib import SequenceMatcher

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
        print("\n\nDevice Used:",self.device)



    def load_model(self):
        return torch.hub.load('yolov5', model='custom', path='LicensPlateModel2.pt', source='local', force_reload=True)


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
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on the frame.
        Only boxes with width and height greater than or equal to the specified minimum values will be plotted.
        :param results: Contains labels and coordinates predicted by the model on the given frame.
        :param frame: Frame which has been scored.
        :param min_width: Minimum width threshold for boxes.
        :param min_height: Minimum height threshold for boxes.
        :return: Frame with qualifying bounding boxes and labels plotted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        bboxes = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bboxes.append((x1,y1,x2,y2))
                width, height = x2 - x1, y2 - y1
                if width >= 100:
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, str(row[4]), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame,bboxes

    def text_similarity(self, list1, list2):
    # Initialize a variable to keep track of the total similarity ratio
        total_similarity = 0.0
        
        # Iterate through the components of the lists
        for text1, text2 in zip(list1, list2):
            # Create a SequenceMatcher object for each pair of components
            seq_matcher = SequenceMatcher(None, text1, text2)
            
            # Get the similarity ratio for the pair of components
            similarity_ratio = seq_matcher.ratio()
            
            # Add the similarity ratio to the total
            total_similarity += similarity_ratio
        
        #ignore the division by zero
        if len(list1) == 0:
            return 0
        # Compute the average similarity ratio

        average_similarity = total_similarity / len(list1)
        
        # Return the average similarity ratio
        return average_similarity
    def OCR(self, frame):
        reader = easyocr.Reader(['ar'])
        return reader.readtext(frame, detail=0)
    def enhance_black(self,image):
        # Convert the image to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance black color by decreasing the intensity
        enhanced = cv2.multiply(grayscale, 0.5)

        return enhanced
    
    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame = cv2.resize(frame, (308, 308))
            frame2 = cv2.resize(frame, (308, 308))
            results = self.score_frame(frame)
            img, bboxes = self.plot_boxes(results, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                print("done1")
                for bbox in bboxes:
                    roi = frame2[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    roi = self.enhance_black(roi)
                    # cv2.imshow("roi", roi)
                    cv2.imwrite("p_pic/roi.jpg", roi)
            
            elif key == ord('v'):
                print("done2")
                texts = []
                for bbox in bboxes:
                    roi = frame2[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    roi = self.enhance_black(roi)
                    cv2.imwrite("p_pic/roi2.jpg", roi)
                    final_img = cv2.imread("p_pic/roi.jpg")
                    final_img2 = cv2.imread("p_pic/roi2.jpg")
                    text = self.OCR(final_img)
                    text2 = self.OCR(final_img2)

                    texts.append(text)
                    texts.append(text2)

                sorted_texts = sorted(texts)
                similarity = self.text_similarity(sorted_texts[0], sorted_texts[1])
                print(f"Text similarity: {similarity}")


            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("img", frame)

            if key == ord('q'):
                break
# Create a new object and execute.
detection = ObjectDetection()
detection()

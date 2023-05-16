import torch
import numpy as np
import cv2
import easyocr
from difflib import SequenceMatcher


class ObjectDetection:

    
    def __init__(self):

        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)



    def load_model(self):
        return torch.hub.load('yolov5', model='custom', path='LicensPlateModel2.pt', source='local', force_reload=True)



    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                #print the threshold value
                # cv2.putText(frame, str(row[4]), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr2, 2)
        return frame, bboxes
    


    def OCR(self, frame):
        reader = easyocr.Reader(['ar'])
        return reader.readtext(frame, detail=0)
    

    # def text_similarity(self, text1, text2):
    #     # Create a SequenceMatcher object
    #     seq_matcher = SequenceMatcher(None, text1, text2)

    #     # Get the similarity ratio
    #     similarity_ratio = seq_matcher.ratio()

    #     # Return the similarity ratio
    #     return similarity_ratio
    
    

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

    def enhance_black_color(self,img):
        # Load the image
        
        
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

    def __call__(self):

        # read pictures
        def predict(img_path,saved_path,visualize=False):
            img = cv2.imread(img_path)
            
            img_resize = cv2.resize(img, (308, 308))
            results = detection.score_frame(img)
            img , bboxes = detection.plot_boxes(results, img_resize)

            for bbox in bboxes:
                roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                roi = self.enhance_black_color(roi)
                if visualize:
                    cv2.imshow(saved_path, roi)
                
                cv2.imwrite(saved_path, roi)

                final_img = cv2.imread(saved_path)
                text = self.OCR(final_img)
                print(f'this is the result of ocr for{img_path}: {text}')

            if visualize:
                cv2.imshow("OpenCV/Numpy normal", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return text
        

        text1 = predict('pic/CarsLP/rescaled/5.jpg','savedpic/roi.jpg',visualize=True)
        text2 = predict('pic/CarsLP/rescaled/6.jpg','savedpic/roi2.jpg',visualize=True)

        print(f'this is the result of text similarity {self.text_similarity(text1,text2)}')


# Create a new object and execute.
detection = ObjectDetection()
detection()

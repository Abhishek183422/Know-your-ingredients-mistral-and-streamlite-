import cv2
import easyocr
import  matplotlib.pyplot as plt
import streamlit as st
import numpy as np

class OcrDetect:
    def __init__(self,image_path):
        self.image_path = image_path
        self.reader = None
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            st.error(f"Failed to initialize EasyOCR Reader: {e}")
            raise

    
    def ocr_detect(self):
        # Load the image
        gray_image = cv2.cvtColor(self.image_path, cv2.COLOR_BGR2GRAY)
    # turn the image in to binary and the thresold is 200,230
        # Resize the image for better OCR accuracy
        resized_image = cv2.resize(gray_image, None, fx=3, fy=4, interpolation=cv2.INTER_LINEAR)

        # Apply CLAHE to improve contrast in shadowed areas
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(20, 20)) #4.0, 15,15
        clahe_image = clahe.apply(resized_image)
        st.image(gray_image,channels='RGB',width=500)
        # Perform OCR on the image
        result = reader.readtext(clahe_image)
        threshold_val_probabi= 0.6 ######3 to be put in config file

        # Print the extracted text
        for (bbox, text, prob) in result:
            if prob >= threshold_val_probabi:
                pass
                # print(f"Detected text: {text} (Probability: {prob:.2f}) fe")

        detected_text=[text for (bbox,text,prob) in result if prob >= threshold_val_probabi]   
        return detected_text


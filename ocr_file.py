import cv2
import easyocr
import  matplotlib.pyplot as plt
import streamlit as st
import numpy as np

reader = easyocr.Reader(['en'])

class OcrDetect:
    def __init__(self,image_path):
        self.image_path=image_path
    
    def ocr_detect(self):
        # Load the image
        gray_image = cv2.cvtColor(self.image_path, cv2.COLOR_BGR2GRAY)
    # turn the image in to binary and the thresold is 200,230
        # Resize the image for better OCR accuracy
        resized_image = cv2.resize(gray_image, None, fx=3, fy=4, interpolation=cv2.INTER_LINEAR)

        # Apply CLAHE to improve contrast in shadowed areas
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(15, 15))
        clahe_image = clahe.apply(resized_image)
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


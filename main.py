from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from ocr_file import OcrDetect
import streamlit as st
import cv2
import numpy as np

from huggingface_hub import InferenceClient


load_dotenv()
##api key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# Streamlit web interface
st.title('KNOW _YOUŘ_ :blue[INGREDIENTS] 🩺👨🏻‍⚕️')
# Create a file uploader widget
st.write("Upload Your ingredient picture here:")
pic_upload_file = st.file_uploader("Choose a file")
st.write("Hello My Name is Dhara your AI Nutritionist. How Can I Help You Today! 👩🏼‍🦱")

class LanchainWork:
    def __init__(self, pic_upload_file):
        self.conversation_history = []
        self.output_parser = StrOutputParser()
        self.pic_uploaded_file = pic_upload_file
        self.ocr_result = None  # initialization as none
        self.ingredient_list = []
        self.mistral_result=[]
        self.client = InferenceClient(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            token="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",)
        self.user_here_input=None 
        
    # creating a session state on st for displaying the user and AI text (chat history display)
    def clear_input_field(self):
        pass  # for now

    # Streamlit interface code and image preprocessing using OpenCV
    def streamlit_view(self):
        # Check if a file was uploaded
        if self.pic_uploaded_file is not None:
            # Display the uploaded picture
            # st.image(self.pic_uploaded_file, channels='RGB', width=500)
            image = np.array(bytearray(self.pic_uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # creating OCR object from ocr_file.py and we imported it here
            ocr = OcrDetect(image)
            # getting the text after OCR detection
            self.ingredient_list = ocr.ocr_detect()
        return self.ingredient_list

    # Mistral chat history part
    def update_conversation_history(self, role, content):
        self.conversation_history.append((role, content))


    def create_prompt(self):
        messages = [("system", "You are a helpful assistant. Please respond to the user queries.")]
        messages.extend(self.conversation_history)
        messages.append(("user", "{question}"))
        return ChatPromptTemplate.from_messages(messages)
    
    def mistral_chat(self,user_input=""):
        response_text = ""
        prompt = self.create_prompt().format(question=user_input)
        

        for message in self.client.chat_completion(
            messages=[{"role": "user", "content":prompt}],
            max_tokens=700,
            stream=True,
            ):

            response_text += message.choices[0].delta.content
        
        return response_text

    def mistral_chat_history(self):
        ## Initialize conversation history
        # Providing the ingredient list here
        if self.ingredient_list:
            self.update_conversation_history("user", self.ingredient_list)
            self.update_conversation_history("user", "Analyse the 'ingredient' or 'Nutrition Facts' present in the list and act as a food pharmacist who answers questions asked by the user regarding the 'ingredient' or 'Nutrition Facts'. Only answer questions related to the ingredient list and food. Don't give long answers; be simple and specific. after seeing the ingredient list do suggest some common question to the user for intializing conversation")
        # Get suggestions for initial questions from the model
        if self.ingredient_list:
            suggested_questions = self.mistral_chat("Can you suggest some common questions related to the ingredient list? and only provide me the questions and nothing else or extra sentence")
            st.write("Suggested Questions:")
            st.write(suggested_questions)

        user_input = st.text_input("How can i help you i am your Ai : ")  # Create text input field
        user_enter_input= st.button("submit")
        if user_enter_input and user_input:
            # Update the conversation history with the user's question
            self.update_conversation_history("user", user_input)

            # Get the model's response
            result_mistral = self.mistral_chat(user_input)

            # Update the conversation history with the assistant's response
            self.update_conversation_history("assistant", result_mistral)

            st.write(result_mistral)


if pic_upload_file is not None:
    app = LanchainWork(pic_upload_file)
    app.streamlit_view()
    app.mistral_chat_history()

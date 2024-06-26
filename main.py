from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 
import os
from dotenv import load_dotenv
from ocr_file import OcrDetect
import streamlit as st
import cv2
import numpy as np

load_dotenv()
##api key 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

#streamlite webinterface
st.title('KNOW _YOUŘ_ :blue[INGREDIENTS] 🩺👨🏻‍⚕️')
# Create a file uploader widget
st.write("Upload Your ingrident picture here :")
pic_upload_file=st.file_uploader("Choose a file")
class lanchain_work:

    def __init__(self,pic_upload_file):
        
        self.conversation_history=[]
        self.llm = Ollama(model="Mistral")
        self.output_parser=StrOutputParser()
        self.pic_uploaded_file = pic_upload_file
        self.ocr_result=None #initilisation as none
        self.ingridient_list = []
    
    # creating a session state on st for displaying the user and AI text. (chat history display)
    def clear_input_field():
        pass # for now
    # streamlite interface code and image preprocessing using opencv
    # needs to add more code img preprocessing code to increase the accuracy
    def streamlit_view(self):
        # Check if a file was uploaded
        if self.pic_uploaded_file is not None:
            # Display the uploaded picture
            st.image(self.pic_uploaded_file,channels='RGB',width=500)
            image = np.array(bytearray(self.pic_uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            #creating ocr object from ocr_file.py and we imported it here
            ocr=OcrDetect(image)
            # getting the text after ocr detection
            #self.ocr_result = ocr.ocr_detect()
            self.ingridient_list = ocr.ocr_detect()
        return self.ingridient_list

    ## mistray chat history part
    def update_conversation_history(self, role, content):
        self.conversation_history.append((role, content))
    
    def create_prompt(self):
        messages = [("system", "You are a helpful assistant. Please respond to the user queries.")]
        messages.extend(self.conversation_history)
        messages.append(("user", "{question}"))
        return ChatPromptTemplate.from_messages(messages)

    
    def mistral_chat(self):
        ## Initialize conversation history
        # Providing the ingredient list here
        if self.ingridient_list:
            self.update_conversation_history("user", self.ingridient_list)
            self.update_conversation_history("user", "Analyse the 'ingridient' or 'Nutrition Facts' presnet in the list and  Act as a food pharmacist who answer question asked by the user regarding the 'ingridient' or 'Nutrition Facts' and only asnwer question related to ingrident list and food and dont give long answer be simple and specific ")
        
        user_input = st.text_input("Type your message here:")
        if user_input:
            if user_input.lower() == "q":
                st.write('HAPPY TO HELP YOU ! BYE 👨🏻‍⚕️')
                return
            # Update the conversation history with the user's question
            self.update_conversation_history("user", user_input)
            
            # Create the prompt with the updated conversation history
            prompt = self.create_prompt()

            # Print the conversation history for debugging purposes
            print("\nConversation History: ")
            for role, message in self.conversation_history:
                print(f"{role}: {message}")

            # Create the chain with the updated prompt
            chain = prompt | self.llm | self.output_parser
            
            # Get the model's response
            response = chain.invoke({"question": user_input})
            
            # Update the conversation history with the assistant's response
            self.update_conversation_history("assistant", response)
            
            # Display the response
            st.write(response)
                      
if pic_upload_file is not None:
    app=lanchain_work(pic_upload_file)
    app.streamlit_view()
    app.mistral_chat()
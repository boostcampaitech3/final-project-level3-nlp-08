from service.front_function import txt_to_json

import requests
import json

import numpy as np
from PIL import Image
import streamlit as st

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

def preprocess(js_file):
    dialogue = js_file['data'][0]['body']['dialogue']
    
    return_string = ""
    for string in dialogue:
        return_string += string['participantID'] + ": " + string['utterance'] + " \r\n "

    return return_string

def main():
    st.title("Golden summary & Show image")

    uploaded_file = st.file_uploader("Input your dialogue data", type=["txt"])
    print(uploaded_file)
    
    if uploaded_file:
        dialogue_data = preprocess(txt_to_json(uploaded_file))  # str

        data = {'dialogue': dialogue_data}

        a = requests.post('http://127.0.0.1:8000/upload', data=json.dumps(data))
        st.write(a)

        image_array = a.json()["image_array"]
        image_array = np.array(image_array)
        converted_image_array = 255 - (image_array * 255).astype(np.uint8)
        image = Image.fromarray(converted_image_array)
        st.write(a.json()["summary"])
        st.image(image, caption='Uploaded Image')


main()
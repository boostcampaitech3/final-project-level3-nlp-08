import streamlit as st

import io
import os
import yaml
import requests
import json

from fastapi import FastAPI

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

def preprocess(upload_file):
    
    dict_data = upload_file['data'][0]['body']['dialogue']
    
    return_string=""
    for string in dict_data:
        return_string += string['participantID'] + ": " + string['utterance'] + "\r\n"
    
    return return_string

def main():
    st.title("Mask Classification Model")
    
    uploaded_file = st.file_uploader("Choose an image", type=["json"])

    if uploaded_file:
        json_data = json.load(uploaded_file)

        data = {'dialogue':preprocess(json_data)}
       
        a = requests.post('http://127.0.0.1:8000/upload', data = json.dumps(data))
    
        st.write(a.json())


main()

import streamlit as st
import pandas as pd
import numpy as np
import time

import io
import os
import yaml
import json

from PIL import Image

from show_img import txt_to_img
from show_img import get_image_download


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


def show_json(file):
    """
    대충 dialog 요약해주는 함수
    """
    with open(str(file), "r", encoding="UTF-8") as js:
        js = json.load(js)
        # 요약해주는 함수를 통해 summary를 가져오게 할거임
        summary = js['body'][4]['context']
        return summary
    # json = json.load(file)
    # st.write(json)


def main():
    st.title("Golden summary & Show image")

    upload_file = st.file_uploader("choose file", type=["json"])  # 먹일 카톡 데이터
    upload_file

    if upload_file:
        golden_summary = show_json(upload_file.name)
        st.write(f'summary:{golden_summary}')
        result_img = txt_to_img(golden_summary)
        # 지금은 미리 저장해놓은 이미지를 불러오지만 나중에는 dalle 이용한 이미지를 가져오게 할것
        st.image(result_img)

        filename='dalle_img'
        st.markdown(get_image_download(result_img, filename +'.png'), unsafe_allow_html=True)

main()
from service.front_function import txt_to_json

import requests
import json

import numpy as np
from PIL import Image
import streamlit as st

from io import BytesIO

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

@st.cache()
def preprocess(js_file):
    dialogue = js_file['data'][0]['body']['dialogue']

    return_string = ""
    for string in dialogue:
        return_string += string['participantID'] + ": " + string['utterance'] + " \r\n "

    return return_string


def main():
    st.title("Golden summary & Show image")

    uploaded_file = st.file_uploader("Input your dialogue data", type=["txt"])

    if uploaded_file:
        dialogue_data = preprocess(txt_to_json(uploaded_file))  # str
        data = {'dialogue': dialogue_data}

        images = []

        with st.spinner("요약문 생성중..."):
            a = requests.post('http://127.0.0.1:8000/upload', data=json.dumps(data))

            summary = a.json()['kor_sum']

            st.write(summary)

        with st.spinner("사진 생성중..."):
            data2 = {'dialogue': summary}
            a = requests.post('http://127.0.0.1:8000/image', data=json.dumps(data2))

            image_arrays = b.json()["image_array"]

            for image_array in image_arrays:
                image_array = np.array(image_array)
                converted_image_array = (image_array * 255).astype(np.uint8)
                image = Image.fromarray(converted_image_array)
                images.append(image)
                st.image(image, caption='Uploaded Image')

        # 생성한 이미지 개수만큼 selectbox 생성
        multi_select = st.multiselect('select image you want', ['picture' + str(i) for i in range(1, len(images) + 1)])

        # 고른 이미지별 다운로드버튼 생성
        for i in range(len(multi_select)):
            img = images[int(multi_select[i][-1]) - 1]
            buf = BytesIO()
            img.save(buf, format="PNG")
            byte_img = buf.getvalue()

            st.download_button(label="Download " + multi_select[i], data=byte_img, file_name=multi_select[i] + '.png')


main()
import streamlit as st
from PIL import Image
import numpy as np

from io import BytesIO
import base64

"""
golden summary를 이미지로 바꿔줄 함수
"""

def txt_to_img(text):
    # st.write(f'your input:{text}')
    # st.write('Extract Image... w8')
    # image = Image.open('select_character.png')
    # st.image(image, caption='Uploaded Image')

    # 대충 이미지 생성하는 코드 -> 입력: text, 리턴: img
    # to_show_img = generator.tensor_to_img(img): (Image.fromarray(torch.cat(img_list, dim=-2).detach().cpu().numpy().astype(np.uint8))
    # to_show_img.save('./generated.jpg')


    image = Image.open('select_character.png')  # streamlit_test 폴더에 이미지를 추가해 설정해주면 됩니다.
    np_array = np.array(image)
    to_show_img = Image.fromarray(np_array)
    return to_show_img


def get_image_download(img,filename):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    byte_im = buffered.getvalue()
    img_str = base64.b64encode(buffered.getvalue()).decode()

    ans = st.download_button(label="Download image", data=byte_im, file_name=filename)
    return ans
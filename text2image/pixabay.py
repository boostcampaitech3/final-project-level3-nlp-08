# 1. import library

import seleniu기
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait

from urllib.request import Request, urlretrieve, URLopener
from time import sleep
import os

# prepare for crawling

def prepare():
    # input theme and make link for crawling

    theme = input("검색어를 입력하세요: ")
    url = f'https://pixabay.com/ko/{theme}/search/' # 한국어 검색어에 대한 url 생성

    # connect to webdriver (by ChromeDriverManager)

    from webdriver_manager.chrome import ChromeDriverManager

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url=url)

    # make folder for specific theme
    # 현재 위치에 img_data/{theme} 폴더 생성

    folder_name = './img_data'           # 상위 폴더(img_data) 생성
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    folder_name = f"./img_data/{theme}"  # 하위 폴더(theme) 생성
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

# get image url

def get_img_url():
    try:
        images = driver.find_elements(by=By.CSS_SELECTOR, value=".container--HcTw2 > div > div > div > div > a > img")
    except:
        print('드라이버 css 찾기 실패.')
    img_url = []
    for image in images:
        try:
            url = image.get_attribute('src')
            img_url.append(url)
        except:
            print(f'{image}_src 얻기 실패.')
    return img_url

# get png file

def get_png_file():
    
    page_num = int(driver.find_element(by=By.XPATH, value='//*[@id="app"]/div[2]/div[4]/div[2]/div[2]/div/span[1]').text[1:])
    print(f"<{theme}> 의 검색 결과 페이지 수 : ", page_num)

    headers={'User-Agent': 'Mozilla/5.0'}
    for i in range(1, page_num+1):
        img_url = get_img_url()
        print(f'{i}_page에서 찾은 이미지 개수 : {len(img_url)}')
        for link in img_url :
            file_name = link.split('/')[-1].split('.')[-2]
            os.system(f'curl {link} > {folder_name}/{file_name}.png')
        try:
            driver.find_element(by=By.XPATH, value='//*[@id="onetrust-close-btn-container"]/button').click() # 팝업창 뜰 시에 창 닫기
        except:
            pass
        driver.find_element(by=By.XPATH, value='//*[@id="app"]/div[2]/div[4]/div[1]/div[2]/a').click()
        sleep(3)

def main():
    prepare()
    get_png_file()

if __name__=="__main__":
    main()
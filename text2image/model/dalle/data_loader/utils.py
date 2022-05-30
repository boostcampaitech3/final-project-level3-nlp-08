import os
import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd





def seed_everything(seed):
    """
    동일한 조건으로 학습을 할 때, 동일한 결과를 얻기 위해 seed를 고정시킵니다.

    Args:
        seed: seed 정수값
    """
    torch.manual_seed(seed)  # pytorch의 random seed 고정
    torch.cuda.manual_seed(seed)  # GPU 에서 사용하는 난수 생성 시드 고정
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # CuDNN 부분고정
    torch.backends.cudnn.benchmark = False  # CuDNN 부분고정
    np.random.seed(seed)  # Numpy 부분
    random.seed(seed)  # transforms에서 random 라이브러리를 사용하기 때문에 random 라이브러리를 불러서 고정

def search(dirname, result):  # 하위목록의 모든 파일을 찾는 함수
    try:
        filenames = os.listdir(dirname)
        print(f'file 개수 : {len(filenames)}')
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                if full_filename.startswith('.'):
                    continue
                search(full_filename, result)
            else:
                ext = os.path.splitext(full_filename)[-1]  # 확장자 체크
                if ext:
                    result.append(full_filename)
                else:
                    print(full_filename)
    except PermissionError:
        print('error')

def labeling(dirname, result, prefix):  # 라벨링하는 함수
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            if filename.startswith('.'):
                continue
            tmp_str = filename.split(".")[0].split("__")[0]
            keyword = prefix + ' '.join(tmp_str.split("-")[:-1])
            result.append(keyword)
    except PermissionError:
        print('error')

def check_unopen_img(df):
    for index in range(len(df)):
        try:
            image = Image.open(df['path'].iloc[index]).convert('RGB')
            label = df['label'].iloc[index]
        except e:
            print(e)
            print(f'Image load error : {df["path"].iloc[index]}')

def rm_unopen_img(df):
    for index in range(len(df)):
        try:
            image = Image.open(df['path'].iloc[index]).convert('RGB')
            label = df['label'].iloc[index]
        except:
            print(f'Image load error : {df["path"].iloc[index]}')
            os.remove(df["path"].iloc[index])

def mk_dataframe(seed):
    seed_everything(seed)
    illust_all_path = []
    search("./img_data/illustrations", illust_all_path)
    scenery_all_path = []
    search("./img_data/scenery", scenery_all_path)
    vector_all_path = []
    search("./img_data/vectors", vector_all_path)
    dirname, illust_label = "./img_data/illustrations", []
    labeling(dirname, illust_label, "an illustration image of ")
    dirname, vector_label = "./img_data/vectors", []
    labeling(dirname, vector_label, "a vector image of ")
    dirname, scenery_label = "./img_data/scenery", []
    labeling(dirname, scenery_label, "a scenery of ")

    illust_df = pd.DataFrame(illust_all_path, columns=['path'])
    illust_df['label'] = illust_label

    scenery_df = pd.DataFrame(scenery_all_path, columns=['path'])
    scenery_df['label'] = scenery_label

    vector_df = pd.DataFrame(vector_all_path, columns=['path'])
    vector_df['label'] = vector_label

    df = pd.concat([illust_df, vector_df, scenery_df], ignore_index=True)

    # if you want to check or remove wrong images, run below method. Then rerun mk_dataframe.
    # check_unopen_img(df)
    # rm_unopen_img(df)
    return df

def data_setting(df):
    # mk Custom dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize
            ]),
        'val':
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize
            ]),
    }

    train, valid = train_test_split(df, test_size=0.2,
                                    shuffle=True,
                                    random_state=42)
    
    return data_transforms, train, valid
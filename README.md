# CALL-E

## 1. About Us

### Members

임동진|정재윤|조설아|허치영|이보림|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/72785706?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/71070496?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/90924434?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/69616444?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/55435898?v=4' height=80 width=80px></img>|
[Github](https://github.com/idj7183)|[Github](https://github.com/kma7574)|[Github](https://github.com/jarammm)|[Github](https://github.com/mooncy0421)|[Github](https://github.com/bo-lim)

### Contribution  

`임동진` &nbsp; : &nbsp; Dialogue Summarization • Backend • Product Serving <br>
`정재윤` &nbsp; : &nbsp; Dataset Processing • Frontend <br>
`조설아` &nbsp; : &nbsp; Text-to-Image • Text processing <br>
`허치영` &nbsp; : &nbsp; Dialogue Summarization • Documentation <br>
`이보림` &nbsp; : &nbsp; Text-to-Image • Image Dataset <br>
<br>

## 2. Project

채팅 로그 요약문으로 단체카톡방 대표 이미지를 생성해주는 ChatDALL-E

### Docs
* [Presentation](https://github.com/boostcampaitech3/final-project-level3-nlp-08/blob/main/assets/NLP_08_CALL-E.pdf)
* [Wrapup Report](https://github.com/boostcampaitech3/final-project-level3-nlp-08/blob/main/assets/%EC%B5%9C%EC%A2%85%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(08%EC%A1%B0).pdf)

### WHY?
- 각 채팅방의 특징을 잘 표현하는 대표 이미지 생성

- 채팅방 혼동으로 인해 의도와 다른 채팅방에 메시지를 잘못 전송하는 경우를 방지하기 위함


### HOW?
- 최근 대화 내역의 요약 문장 제공

- 최근 대화 내역 요약 문장을 나타내는 이미지 제공


### Architecutre
![image](https://github.com/boostcampaitech3/final-project-level3-nlp-08/blob/main/assets/Architecture%20block%20diagram.png?raw=true)


## 3. 시연 영상
![최종-데모-영상 (1)](https://user-images.githubusercontent.com/69616444/172797376-e849ba3a-a633-48f2-bd6d-a05d4fef3ca2.gif)


## 4. Structure
```
.
├── summarization - model for dialogue summarization
│   └── ...
├── text2image - model for text-to-image (minDALL-E)
│   └── ...
|-- README.md
|-- app.py - Frontend Service
|-- main.py - Backend Service
|-- requirements.txt
|-- client.json - Papago API ID & Key
└── service - methods for frontend/backend
    |-- __init__.py
    |-- back_function.py  - Backend Method
    |-- front_function.py - Frontend Method
    |-- text_to_image.py  - Text to Image Model Method
    └── utils             - Modules required to load the minDALL-E
        |-- __init__.py
        |-- config.py
        |-- vqgan_layer.py
        |-- transfomer_layer.py
        |-- sampling.py
        |-- tokenizer.py
        |-- transformer.py
        |-- utils.py
        └── vqgan.py

```


## 5. Usage
### Installing required libraries
* `pip install -r requirements.txt --use-feature=2020-resolver`
  * use-feature : Library dependency resolver를 위한 argument

### Run Frontend-File
* `streamlit run app.py`

### Run Backend-File
* `uvicorn main:app`

### Text to Image Model
* text2image/model/tf_model/model/

  * Fine-tuning 시킨 minDALL-E의 Parameter 저장

* service/

  * Service에 활용할 minDALL-E Parameter 저장 공간
     
### Dialogue Summarization Model
* Python
```python
from transformers import BartForConditionalGeneration
model_name = 'chi0/kobart-dial-sum'
model = BartForConditionalGeneration.from_pretrained(model_name)
```

* Huggingface Model 

  * https://huggingface.co/chi0/kobart-dial-sum

## 6. Product Serving
### 활용한 Cloud
- Google Cloud Platform
  - 활용 이유 1 : 무료로 GPU를 활용하기에 최적인 Cloud Platform
  - 활용 이유 2 : Machine Learning을 위한 환경을 자동으로 구성해주는 기능 존재 

### HARDWARE
- Machine 유형 : n1-standard-4
  - vCPU : 4
  - Memory : 15GB
  - CPU 플랫폼 : Intel Haswell 
  - File 시스템 공간
  
  ![image](https://user-images.githubusercontent.com/72785706/172453568-f37cbd97-7b94-49e3-9f88-ae4347f69c0e.png)

- GPU : NVIDIA Tesla T4 1개
  - Tesla T4 사양 : https://www.nvidia.com/ko-kr/data-center/tesla-t4/

### Environment
- HTTP 및 HTTPS 트래픽 활용

- Port : 8501을 제외한 모든 포트는 외부에서 통과 불가능하도록 설정

## Cloud에 서비스 활용 방법
0. `pip install -r requirements.txt --use-feature=2020-resolver`
   * pip이 설치 되어 있지 않다면 pip 설치가 선행되어야 

1. service directory, main.py, app.py를 Cloud에 Load함

2. Fine-Tuning 시킨 Text to Image 모델에 대해 저장된 State들을 service 폴더 아래에 저장함

3. main.py의 `txt2imgModel,_ = Rep_Dalle.from_pretrained({HERE})`부분에서 {HERE}에 Model State를 저장한 경로를 넣어줌

4. Papago API에서 번역 API를 신청한 후, API ID 및 Private Key를 client.json 파일에 넣어줌
   * client.json 형식
    ```
    {
        "client_id":{Papago API ID},
        "client_secret":{Papago API Key}
    }
    ```

5. `nohup uvicorn main:app &` <br> `nohup streamlit run app.py &`
   * Cloud에서 app.py 및 main.py를 실행시킴으로써 24시간 사이트를 활용할 수 있도록 배포하면 됨
   * 최종 Product Servin Site : `{Cloud IP 주소}:8501`

## 7. References
### Datasets
- [AI Hub 한국어 대화 요약 데이터셋](https://aihub.or.kr/aidata/30714)
- [Pixabay](https://pixabay.com/ko/)
- [KorSTS](https://github.com/kakaobrain/KorNLUDatasets)

### Papers
- [Ham, Jiyeon, et al. "Kornli and korsts: New benchmark datasets for korean natural language understanding." arXiv preprint arXiv:2004.03289 (2020).](https://arxiv.org/pdf/2004.03289.pdf)
- [Lewis, Mike, et al. "Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension." arXiv preprint arXiv:1910.13461 (2019).](https://arxiv.org/pdf/1910.13461.pdf)
- [Liu, Zhengyuan, and Nancy F. Chen. "Controllable Neural Dialogue Summarization with Personal Named Entity Planning." arXiv preprint arXiv:2109.13070 (2021).](https://arxiv.org/pdf/2109.13070.pdf)
- [Lee, Dongyub, et al. "Reference and document aware semantic evaluation methods for Korean language summarization." arXiv preprint arXiv:2005.03510 (2020).](https://arxiv.org/pdf/2005.03510.pdf)
- [Ramesh, Aditya, et al. "Zero-shot text-to-image generation." International Conference on Machine Learning. PMLR, 2021.](https://arxiv.org/pdf/2102.12092.pdf)
- [Zhou, Yufan, et al. "LAFITE: Towards Language-Free Training for Text-to-Image Generation." arXiv preprint arXiv:2111.13792 (2021).](https://arxiv.org/pdf/2111.13792.pdf)
- [Rombach, Robin, et al. "High-Resolution Image Synthesis with Latent Diffusion Models." arXiv preprint arXiv:2112.10752 (2021).](https://arxiv.org/pdf/2112.10752.pdf)

### Models
- [kakaobrain/minDALL-E](https://github.com/kakaobrain/minDALL-E)
- [LAFITE](https://github.com/drboog/Lafite)
- [Latent-Diffusion](https://github.com/CompVis/latent-diffusion)
- [STK-AI/KoBART](https://github.com/SKT-AI/KoBART)
- [AIRC-KETI/KE-T5](https://github.com/AIRC-KETI/ke-t5)

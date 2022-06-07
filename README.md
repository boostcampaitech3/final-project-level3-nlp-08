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

### WHY?
- 각 채팅방의 특징을 잘 표현하는 대표 이미지 생성

- 채팅방 혼동으로 인해 의도와 다른 채팅방에 메시지를 잘못 전송하는 경우를 방지하기 위함


### HOW?
- 최근 대화 내역의 요약 문장 제공

- 최근 대화 내역 요약 문장을 나타내는 이미지 제공


### Architecutre
![image](https://github.com/boostcampaitech3/final-project-level3-nlp-08/blob/main/assets/Architecture%20block%20diagram.png?raw=true)


## 3. 시연 영상
-- 영상 -- 


## 4. Structure
```
.
├── service - methods for frontend/backend
│   └── ...
├── summarization - model for dialogue summarization
│   └── ...
├── text2image - model for text-to-image (minDALL-E)
│   └── ...
├── .gitignore
├── README.md
├── app.py - Frontend Service
├── main.py - Backend Service
└── requirements.txt
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
### HARDWARE

### HOW?
[임동진 TODO]


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

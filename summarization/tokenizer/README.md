# Custom Tokenizer
## Error
```python
Traceback (most recent call last):
  File "misc.py", line 23, in <module>
    dummy_tokenizer.save_pretrained("./kobart_tokenizer/")
  File ".../venv/lib/python3.6/site-packages/transformers/tokenization_utils_base.py", line 1992, in save_pretrained
    filename_prefix=filename_prefix,
  File ".../venv/lib/python3.6/site-packages/transformers/tokenization_utils_fast.py", line 535, in _save_pretrained
    vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
  File ".../venv/lib/python3.6/site-packages/transformers/tokenization_utils_base.py", line 2044, in save_vocabulary
    raise NotImplementedError
NotImplementedError
```

<br>

## Error 해결에 대한 링크
* [save_pretrained() 에 NotImplemented Error 발생](https://github.com/SKT-AI/KoBART/issues/14)

<br>

## Custom Tokenizer의 필요성
* SKT-AI/KoBART github repository의 위와 같은 문제 발생으로 인해 `save_vocabulary` 구현
* Huggingface에 업로드하기 위해서 vocabulary 파일 저장하기로 결정

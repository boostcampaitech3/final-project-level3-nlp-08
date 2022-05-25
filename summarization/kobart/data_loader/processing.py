def preprocess_function(examples, tokenizer, max_source_length, max_target_length, padding,
                        use_t5:bool = False, prefix:str = None):
    inputs, targets = examples['dialogue'], examples['summary']

    # input에 prefix 추가 (t5 계열 모델만 prefix 사용함)
    # prefix 추가한 input text를 tokenizer 통해 id로 변경 == model_inputs
    if use_t5:
        inputs = [prefix + inp for inp in inputs]

    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    targets = tokenizer(targets, padding=padding, truncation=True, max_length=max_target_length)

    batch = {k:v for k, v in model_inputs.items()}

    batch['labels'] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in targets["input_ids"]
    ]

    return batch


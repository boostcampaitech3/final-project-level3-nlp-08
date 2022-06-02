import nltk

def preprocess_function(examples, tokenizer, max_source_length, max_target_length, padding):
    inputs, targets = examples['dialogue'], examples['summary']

    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    targets = tokenizer(targets, padding=padding, truncation=True, max_length=max_target_length)

    batch = {k:v for k, v in model_inputs.items()}

    batch['labels'] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in targets["input_ids"]
    ]

    return batch

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    # prediction, labels 각 문장 끝에 줄바꿈 붙이기
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def postprocess_text_first_sent(preds):
    preds = [pred.strip() for pred in preds]
    preds = [pred[:pred.index(".")+1] if "." in pred else pred for pred in preds]

    return preds
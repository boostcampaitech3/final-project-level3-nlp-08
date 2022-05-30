from data_loader.processing import preprocess_function

def return_tokenized_sentence(untokenized_sentence, tokenizer, data_args):
    padding = "max_length" if data_args.pad_to_max_length else False

    tokenized_sentence = untokenized_sentence.map(
        lambda example: preprocess_function(examples=example,
                                            tokenizer=tokenizer,
                                            max_source_length=data_args.max_source_length,
                                            max_target_length=data_args.max_target_length,
                                            padding=padding),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=untokenized_sentence.column_names
    )

    return tokenized_sentence
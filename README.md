
tokenizer.sep_token_id = NONE
Первоначально: 
tokenizer.pad_token = NONE


tokenizer.pad_token = tokenizer.eos_token
tokenizer.eos_token = 50256


wget -P ./data/ https://code.s3.yandex.net/deep-learning/tweets.txt

jupyter lab --ip=0.0.0.0 --no-browser
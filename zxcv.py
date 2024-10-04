import string
import pandas as pd

from transformers import AutoTokenizer

from train import define_argparser

# csv = pd.read_csv('./data/final_matched.mod.shuf.train.csv', on_bad_lines='skip', names=['index', 'input', 'output'])
# a = list(csv['output'][:100000])
# a = ' '.join(str(i) for i in a)
# print('애플: {}'.format(a.count('Apple')))
# print('LG: {}'.format(a.count('LG')))
# print('삼성: {}'.format(a.count('삼성')))
# print('캐리어: {}'.format(a.count('캐리어')))


config = define_argparser()
tokenizer = AutoTokenizer.from_pretrained(config.huggingface_model_name)
a = tokenizer.decode([21820])
print(a)
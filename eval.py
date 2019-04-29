import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
from data import data

print("=======TEST.PY IMPORTED WHAT THE FUCK=======")


metadata, idx_q, idx_a = data.load_data(PATH='data/')

w2idx = metadata['w2idx']  # dict  word 2 index
idx2w = metadata['idx2w']  # list index 2 word

print("Loading vocab done:", "shapes", idx_q.shape, idx_a.shape)


emb_dim = 512
batch_size = 256
xvocab_size = yvocab_size = len(idx2w)

unk_id = w2idx['unk']  # 1
pad_id = w2idx['_']  # 0

start_id = xvocab_size
end_id = xvocab_size + 1

w2idx['start_id'] = start_id
w2idx['end_id'] = end_id
idx2w = idx2w + ['start_id', 'end_id']

xvocab_size = yvocab_size = xvocab_size + 2
w2idx['end_id']

print("Vocab preprocessing done")


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import model
print("Start testing")

seq2seq = model.Model(w2idx, idx2w, True)
sess = seq2seq.restore()
#seq2seq.train(trainX, trainY)
questions = [
    'что думаешь об nlp', 
    'кем ты работаешь',
    'какой сегодня день'
]
answers = seq2seq.predict(sess, questions)
new_answers = [seq2seq.predict_one(sess, q) for q in questions]

for q, a, new_a in zip(questions, answers, new_answers):
    print(q)
    print(">", " ".join(a))
    print(">", " ".join(new_a))


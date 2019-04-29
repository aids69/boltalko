import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import model
from data import data


metadata, idx_q, idx_a = data.load_data(PATH='data/')

w2idx = metadata['w2idx']  # dict  word 2 index
idx2w = metadata['idx2w']  # list index 2 word

print("Loading vocab done:", "shapes", idx_q.shape, idx_a.shape)

(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

trainX = trainX.tolist()
trainY = trainY.tolist()
testX = testX.tolist()
testY = testY.tolist()
validX = validX.tolist()
validY = validY.tolist()

print("Split dataset done: ", 'q=', trainX[0], 'a=', trainY[0])

trainX = tl.prepro.remove_pad_sequences(trainX)
trainY = tl.prepro.remove_pad_sequences(trainY)
testX = tl.prepro.remove_pad_sequences(testX)
testY = tl.prepro.remove_pad_sequences(testY)
validX = tl.prepro.remove_pad_sequences(validX)
validY = tl.prepro.remove_pad_sequences(validY)

trainX[0]

# parameters
xseq_len = len(trainX)
yseq_len = len(trainY)

emb_dim = 512
batch_size = 256
n_step = int(xseq_len / batch_size)
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

print("Start testing")

seq2seq = model.Model(w2idx, idx2w, True)
sess = seq2seq.restore()
questions = [
    'что думаешь об nlp',
    'кем ты работаешь',
    'какой сегодня день'
]
answers = seq2seq.predict(sess, questions)
for q, a in zip(questions, answers):
    print(q)
    print(">", " ".join(a))



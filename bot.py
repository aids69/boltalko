import os
from telegram.ext import Filters, MessageHandler, Updater

import bot_config
from data import data
import model

WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя ' # space is included in whitelist
BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
seq2seq, sess = None, None


def echo(bot, update):
    query = update.message.text
    print('Conversation query:', query)
    query = ''.join([c for c in query.lower() if c in WHITELIST])
    reply_ar = seq2seq.predict(sess, [query])[0]
    reply = []
    for w in reply_ar:
        if w == 'end_id':
            break
        reply.append(w)

    answer = ' '.join(reply).capitalize()
    print('Conversation reply:', answer)
    update.message.reply_text(answer)


if __name__ == '__main__':
    # init tf model here
    updater = Updater(token=bot_config.TOKEN)
    updater.dispatcher.add_handler(MessageHandler(Filters.text, echo))
    updater.start_polling()
    # prepare and load model
    metadata, idx_q, idx_a = data.load_data(PATH='data/')
    w2idx = metadata['w2idx']  # dict  word 2 index
    idx2w = metadata['idx2w']  # list index 2 word
    print('Loading vocab done:', 'shapes', idx_q.shape, idx_a.shape)
    vocab_size = len(idx2w)

    start_id = vocab_size
    end_id = vocab_size + 1

    w2idx['start_id'] = start_id
    w2idx['end_id'] = end_id

    idx2w += ['start_id', 'end_id']
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    seq2seq = model.Model(w2idx, idx2w, True)
    sess = seq2seq.restore()

    print('Start listening...')
    updater.idle()

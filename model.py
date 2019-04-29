import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time

class Model:
    def __init__(self, vocab, inv_vocab, forward_only):
        self.ckpt_dir = './checkpoint/'
        self.epochs = 20
        self.batch_size = 256
        self.max_sen_len = 15
        vocab_size = len(vocab)
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.start_id = vocab['start_id']
        self.end_id = vocab['end_id']

        with tf.variable_scope('wat', reuse=tf.AUTO_REUSE):
            self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None, ], name='enc_input_' + str(i)) for i in
                                   range(self.max_sen_len)]
            self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None, ], name='dec_input_' + str(i)) for i in
                                   range(self.max_sen_len + 2)]
            self.decoder_targets = [tf.placeholder(tf.int32, shape=[None, ], name='dec_target_' + str(i)) for i in
                                    range(self.max_sen_len + 2)]
            self.decoder_mask = [tf.placeholder(tf.float32, shape=[None, ], name='dec_mask_' + str(i)) for i in
                                 range(self.max_sen_len + 2)]
            self.forward_only = tf.placeholder(tf.bool, name="forward_only") 

            def create_rnn_cell():
                cell = tf.contrib.rnn.GRUCell(200)
                
                if forward_only:
                    return cell

                dropCell = tf.contrib.rnn.DropoutWrapper(
                    cell,
                    input_keep_prob=1.0,
                    output_keep_prob=0.8
                )

                return dropCell



            outputs, state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                encoder_inputs=self.encoder_inputs,
                decoder_inputs=self.decoder_inputs,
                cell=tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(2)]),
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=512,
                feed_previous=self.forward_only

                #feed_previous=forward_only
            )
            self.outputs = outputs
            
            with tf.name_scope("optimization"):
                self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                    self.outputs,
                    self.decoder_targets,
                    self.decoder_mask
                )

                #self.op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)    
                opt = tf.train.AdamOptimizer(learning_rate=0.001)   

                gradients = opt.compute_gradients(self.loss)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                self.op = opt.apply_gradients(capped_gradients)
        
    def train(self, X, y):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        steps_per_epoch = len(X) // self.batch_size

        for epoch in range(self.epochs):
            print('Epoch', epoch)
            epoch_loss = 0

            t = time.time()

            for b_idx, batch in enumerate(
                    tl.iterate.minibatches(inputs=X, targets=y, batch_size=self.batch_size, shuffle=True,
                                           allow_dynamic_batch_size=True)):
                batch_x = batch[0]
                batch_y = batch[1]
                enc_inputs = tl.prepro.pad_sequences(
                    batch_x, maxlen=self.max_sen_len, padding='pre')
                batch_y_ended = tl.prepro.sequences_add_end_id(
                    batch_y, end_id=self.end_id)
                dec_target = tl.prepro.pad_sequences(
                    batch_y_ended, maxlen=self.max_sen_len + 2)
                dec_inputs = tl.prepro.pad_sequences(
                    tl.prepro.sequences_add_start_id(
                        batch_y, start_id=self.start_id
                    ), maxlen=self.max_sen_len + 2
                )
                dec_mask = tl.prepro.sequences_get_mask(
                    dec_target)

                feed_dict = {}
                feed_dict[self.forward_only] = False
                for i in range(self.max_sen_len):
                    feed_dict[self.encoder_inputs[i]] = [s[i] for s in enc_inputs]
                for i in range(self.max_sen_len + 2):
                    feed_dict[self.decoder_inputs[i]] = [s[i] for s in dec_inputs]
                    feed_dict[self.decoder_targets[i]] = [s[i] for s in dec_target]
                    feed_dict[self.decoder_mask[i]] = [s[i] for s in dec_mask]

                sess.run(self.op, feed_dict=feed_dict)

                if b_idx % 100 == 0:
                    loss = sess.run(self.loss, feed_dict=feed_dict)
                    batch_delta = time.time() - t
                    print(f'Epoch {epoch} [{b_idx}/{steps_per_epoch}] loss: {loss}, time: {batch_delta} ({batch_delta / 100} per/batch)')
                    epoch_loss += loss
                    #self.predict(sess)
                    t = time.time()
            saver.save(
                sess=sess,
                save_path=self.ckpt_dir + 'seq2seq_model.ckpt',
                global_step=epoch)
            print("Epoch loss:", epoch_loss / steps_per_epoch * 10)


    def predict(self, sess, questions):
        mini_batch = []
        for q in questions:
            mini_batch.append([self.vocab.get(w, self.vocab['unk']) for w in q.split()])

        enc_inputs = tl.prepro.pad_sequences(
            mini_batch, maxlen=self.max_sen_len, padding='pre')
        dec_inputs = [self.start_id for _ in mini_batch]
        feed_dict = {}
        feed_dict[self.forward_only] = True
        for i in range(self.max_sen_len):
            feed_dict[self.encoder_inputs[i]] = [s[i] for s in enc_inputs]
        
        feed_dict[self.decoder_inputs[0]] = dec_inputs
        for i in range(1, self.max_sen_len + 2):
            feed_dict[self.decoder_inputs[i]] = dec_inputs
        
        answer = sess.run(self.outputs, feed_dict=feed_dict)

        answers = []
        for idx, q in enumerate(questions):
            answers.append([self.inv_vocab[np.argmax(word[idx])] for word in answer])
        return answers
        


    def restore(self):
        saver = tf.train.Saver(max_to_keep=5)
        sess = tf.Session()
        ckpt=tf.train.get_checkpoint_state(self.ckpt_dir)
        print("================================")
        if ckpt and ckpt.model_checkpoint_path:
            print("Checkpoint restored successfully")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Checkpoint FAILED to restore")
            sess.run(tf.global_variables_initializer())
        print("================================")
        return sess



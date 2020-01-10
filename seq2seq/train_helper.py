import time
import tensorflow as tf
from seq2seq.batcher import batcher
from seq2seq.loss_func import totally_loss
from utils.word2vec_dataload import Vocab

def train_model(model, vocab, params, checkpoint_manager):

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.001)

    @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32)))
    def train_step(enc_input, extended_enc_input,max_oov_len, dec_input,dec_target, enc_pad_mask, padding_mask):


        with tf.GradientTape() as tape:
            # 1. 构建encoder
            enc_output, enc_hidden = model.call_encoder(enc_input)

            # 2. 复制
            dec_hidden = enc_hidden
            # 3. <START> * BATCH_SIZE
            #dec_input = tf.expand_dims([Vocab.START_DECODING] * params['batch_size'], 1)

            predictions, _, coverage, attentions = model(dec_input, enc_output, dec_hidden, extended_enc_input,max_oov_len)

            batch_loss = totally_loss(dec_target, predictions, padding_mask, attentions, 1.0)

            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + \
                        model.attention.trainable_variables


            gradients = tape.gradient(batch_loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss


    for epoch in range(params['epochs']):
        start = time.time()
        dataset = batcher(vocab, params)

        total_loss = 0
        step = 0
        for encoder_batch_data, decoder_batch_data in dataset:

            batch_loss, log_loss, coverage_loss = train_step(encoder_batch_data["enc_input"],
                                    encoder_batch_data["extended_enc_input"],
                                    encoder_batch_data["max_oov_len"],
                                    decoder_batch_data["dec_input"],
                                    decoder_batch_data["dec_target"],
                                    enc_pad_mask=encoder_batch_data["encoder_pad_mask"],
                                    padding_mask=decoder_batch_data["decoder_pad_mask"])

            total_loss += batch_loss

            if step % 25 == 0:
                print('Epoch {} Batch {} batch_loss {:.4f} log_loss {:.4f} coverage_loss {:.4f}'.format(epoch + 1,
                                                                                                        step,
                                                                                                        batch_loss.numpy(),
                                                                                                        log_loss.numpy(),
                                                                                                        coverage_loss.numpy()))

            step += 1
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_manager.save()

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / step))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


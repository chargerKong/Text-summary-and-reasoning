from seq2seq.layer import *
from utils.word2vec_dataload import *
import os
from seq2seq.seq2seq_model import Seq2Seq
from seq2seq.train_helper import train_model
from utils.data_loads import load_test_data
from seq2seq.batcher import batcher

def test(params):
    vo = Vocab()
    params['vocab_size'] = vo.count

    model = Seq2Seq(params)
    ckpt = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    ckpt.restore(checkpoint_manager.latest_checkpoint)


    # save results
    # test_data = load_test_data(params['max_enc_len'])
    test_data = batcher(vo, params)
    # predict test data using beam search
    results = model.model_predict(test_data)[:20000]

    # save the predictions
    model.save_predict_csv(results)



if __name__ == '__main__':
    params = {}
    params["max_enc_len"] = 200
    params["max_dec_len"] = 41
    params["batch_size"] = 256
    params["epochs"] = 4
    params["beam_size"] = 3
    params["embedding_dim"] = 300
    params["enc_units"] = 512
    params["dec_units"] = 512
    params["attention_units"] = 20
    params['learning_rate'] = 1e-4
    params['checkpoints_save_steps'] = 2
    params['max_dec_steps'] = 50
    params['mode'] = 'test'
    params['pointer_gen'] = True
    test(params)


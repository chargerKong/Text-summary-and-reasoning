from seq2seq.layer import *
from utils.word2vec_dataload import load_embedding
from seq2seq.beam_search import beam_decode
from tqdm import tqdm
import pandas as pd
from utils.config import *
from utils.file_opeate import get_result_filename

class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.embedding_matrix = load_embedding()
        self.params = params
        self.encoder = Encoder(params['enc_units'],
                               params["embedding_dim"],
                               self.embedding_matrix,
                               params['batch_size'],
                               params['vocab_size'])

        self.attention = BahdanauAttention(params["attention_units"])

        self.decoder = Decoder(params["vocab_size"],
                               params["embedding_dim"],
                               self.embedding_matrix,
                               params["dec_units"],
                               params["batch_size"])

        self.point = Point()

    def call_encoder(self, inputs):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_state = self.encoder(inputs, enc_hidden)
        return enc_output, enc_state

    def call_decoder_onestep(self, dec_input, enc_output, context_vector, coverage, batch_extend_vocab, batch_oov_len):


        dec_inp, pred, dec_hidden = self.decoder(dec_input,
                                        None,
                                        None,
                                        context_vector)
        context_vector, attention_weights, coverage = self.attention(dec_hidden, enc_output, coverage)
        p_gen = self.point(context_vector, dec_hidden, tf.squeeze(dec_inp, axis=1))
        #rint('has been called once')
        fina_dist = _calc_final_dist(batch_extend_vocab, [pred], [attention_weights], [p_gen], batch_oov_len, self.params['vocab_size'], self.params['batch_size'])
        return fina_dist, dec_hidden, context_vector, attention_weights, coverage

    def __call__(self, dec_input, enc_output, dec_hidden, batch_extend_vocab, batch_oov_len):
        predictions = []
        attentions = []
        p_gens = []
        coverages = []
        coverage = None

        context_vector, attn, coverage = self.attention(dec_hidden, enc_output, coverage)
        for t in range(dec_input.shape[1]):

            dec_inp, pred, dec_hidden = self.decoder(tf.expand_dims(dec_input[:, t], 1),
                                            None,
                                            None,
                                            context_vector)
            # pgn
            p_gen = self.point(context_vector, dec_hidden, tf.squeeze(dec_inp, axis=1))

            context_vector, attn, coverage = self.attention(dec_hidden, enc_output, coverage)

            # using teacher forcing
            #dec_input = tf.expand_dims(dec_target[:, t], 1)

            predictions.append(pred)
            attentions.append(attn)
            p_gens.append(p_gen)
            coverages.append(coverage)

        fina_dist = _calc_final_dist(batch_extend_vocab, predictions, attentions, p_gens, batch_oov_len,
                                     self.params['vocab_size'], self.params['batch_size'])


        # [(32,80000),(32,80000),...m(32,80000),] lens of 33  --> 32 33 80000
        return tf.stack(fina_dist, 1), dec_hidden, coverage, attentions

    def model_predict(self, test_data):
        # save results
        results = []
        pbar = tqdm(total=model.params['vocab_size'] // model.params['batch_size'] + 1)
        for batch_x_data, _ in test_data:
            pbar.update(1)

            batch_data = batch_x_data['enc_input']

            # predict using beam_decode
            results += beam_decode(model, batch_data, params, batch_x_data['extended_enc_input'],
                                   batch_x_data['max_oov_len'])
        return results

    def save_predict_csv(self, results):
        test_df = pd.read_csv(TEST_PATH)
        test_df['Prediction'] = results
        test_df = test_df[['QID', 'Prediction']]

        def submit_proc(sentence):
            sentence = sentence.lstrip(' ，！。')
            sentence = sentence.replace(' ', '')
            if sentence == '':
                sentence = '随时联系'
            return sentence

        test_df['Prediction'] = test_df['Prediction'].apply(lambda x: submit_proc(x))
        result_save_path = get_result_filename(params["batch_size"], params["epochs"], params["max_enc_len"],
                                               params["embed_size"],
                                               commit='beam_search_seq2seq_code')
        test_df.to_csv(result_save_path, index=None, sep=',')


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """
    Calculate the final distribution, for the pointer-generator model
    Args:
    vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                The words are in the order they appear in the vocabulary file.
    attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
    Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """

    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    extended_vsize = vocab_size + batch_oov_len  # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((batch_size, batch_oov_len))
    # list length max_dec_steps of shape (batch_size, extended_vsize)
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
    # then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_vsize]

    # list length max_dec_steps (batch_size, extended_vsize)
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
    # the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                   zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists





if __name__ == '__main__':
    from utils.word2vec_dataload import Vocab
    vocab, reversed_vocab = Vocab.load_vocab()
    params = {}
    params["max_enc_len"] = 200
    params["max_dec_len"] = 41
    params["batch_size"] = 32
    params["epochs"] = 4
    params["beam_size"] = 3
    params["embedding_dim"] = 300
    params["enc_units"] = 512
    params["dec_units"] = 512
    params["attention_units"] = 20
    params['learning_rate'] = 1e-4
    params['checkpoints_save_steps'] = 2
    params['max_dec_steps'] = 50
    params['mode'] = 'train'
    params['pointer_gen'] = True
    params['vocab_size'] = len(vocab)

    model = Seq2Seq(params)
    inputs = tf.ones((params['batch_size'], params['max_enc_len']))
    encoder_output, encoder_state = model.call_encoder(inputs)
    dec_input = tf.expand_dims([1]*params['batch_size'], axis=1)
    dec_targ = tf.ones((params['batch_size'], params['max_enc_len']))
    print(dec_targ.shape)
    model(dec_input, encoder_output, encoder_state, dec_targ,[1],[1])


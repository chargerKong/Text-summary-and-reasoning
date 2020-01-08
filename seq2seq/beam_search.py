import tensorflow as tf
import numpy as np
from utils.word2vec_dataload import Vocab


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, hidden, attn_dists):
        self.tokens = tokens  # list of all the tokens from time 0 to the current time step t
        self.log_probs = log_probs  # list of the log probabilities of the tokens of the tokens
        self.hidden = hidden  # decoder hidden state after the last token decoding
        self.attn_dists = attn_dists  # attention dists of all the tokens
        self.abstract = ""

    def extend(self, top_k_ids, top_k_log_probs, dec_hidden, attention_weights, width_beam):
        """Method to extend the current hypothesis by adding the next decoded token and all the informations associated with it"""
        # 　返回每一个batch中,由于一个字产生的三个最大可能的字
        return [Hypothesis(tokens=tf.concat((self.tokens, top_k_ids[:, i:i + 1]), axis=1),  # we add the decoded token
                           log_probs=tf.concat((self.log_probs, top_k_log_probs[:, i:i + 1]), axis=1),
                           # we add the log prob of the decoded token
                           hidden=dec_hidden,  # we update the state
                           attn_dists=attention_weights) for i in range(width_beam)]

    @property
    def latest_token(self):
        return np.array(self.tokens)[:, -1]

    @property
    def tot_log_prob(self):
        return np.sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def fill_to_batch(data, batch_size):
    shape = data.shape
    fill_data = tf.zeros(shape=(batch_size-shape[0], shape[1]), dtype=data.dtype)
    return tf.concat([data, fill_data], axis=0)


def beam_decode(model, batch, params,batch_extend_vocab,batch_oov_len):
    vocab, reverse_vocab = Vocab.load_vocab()
    # 初始化mask
    start_index = vocab['START']
    stop_index = vocab['STOP']

    batch_size = params['batch_size']

    # 单步decoder
    def decoder_onestep(enc_output, dec_input, context_vector, batch_extend_vocab,batch_oov_len,coverage=None):
        # 单个时间步 运行
        preds, dec_hidden, context_vector, attention_weights,coverage = model.call_decoder_onestep(dec_input, enc_output, context_vector,
                                                                                          coverage, batch_extend_vocab, batch_oov_len)
        # 拿到top k个index 和 概率
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(preds), k=params["beam_size"])
        # 计算log概率
        top_k_log_probs = tf.math.log(top_k_probs)
        # 返回需要保存的中间结果和概率
        return preds, dec_hidden, context_vector, attention_weights, top_k_log_probs, top_k_ids,coverage

    shape = batch.shape
    is_full = True
    if shape[0] != batch_size:
        is_full = False
        batch = fill_to_batch(batch, batch_size)
        batch_extend_vocab = fill_to_batch(batch_extend_vocab, batch_size)


    # 计算第encoder的输出
    enc_output, enc_hidden = model.call_encoder(batch)
    # 计算第一个词预测的结果
    dec_input = tf.expand_dims([start_index] * batch_size, 1)
    dec_hidden = enc_hidden
    context_vector, attention_weights, coverage = model.attention(dec_hidden, enc_output, None)

    preds, dec_hidden, context_vector, attention_weights, top_k_log_probs, top_k_ids, coverage = decoder_onestep(enc_output,
                                                                                                       dec_input,
                                                                                                       context_vector,
                                                                                                       batch_extend_vocab,
                                                                                                       batch_oov_len,
                                                                                                       coverage)
    # 初始化
    initial_hyp = [Hypothesis(tokens=top_k_ids[:, i:i + 1],
                              log_probs=top_k_log_probs[:, i:i + 1],
                              hidden=dec_hidden,
                              attn_dists=attention_weights) for i in range(params['beam_size'])]

    steps = 0
    while steps < params['max_dec_len']:
        hyp_of_nine = []
        for single_hyp in initial_hyp:
            dec_input = tf.expand_dims(single_hyp.latest_token, axis=1)
            dec_hidden = single_hyp.hidden
            preds, dec_hidden, context_vector, attention_weights, top_k_log_probs, top_k_ids, coverage =  decoder_onestep(enc_output,
                                                                                                       dec_input,
                                                                                                       context_vector,
                                                                                                       batch_extend_vocab,
                                                                                                       batch_oov_len,
                                                                                                       coverage)
            hyp_of_nine.extend(
                single_hyp.extend(top_k_ids, top_k_log_probs, dec_hidden, attention_weights, params['beam_size']))

        # 　计算生成某个词的概率总和, 吧log_prob记录的概率想加.
        # 返回一个长度为k^2,(9)的列表.[], 里面的每一个元素为一个batch中, 每一句生成的概率
        # [tf.tensor shape=(32,1), tf.tensor shape=(32,1), tf.tensor shape=(32,1), ..., tf.tensor shape=(32,1),]
        a = [tf.expand_dims(np.sum(hyp_of_nine[i].log_probs, axis=1), axis=1) for i in range(len(hyp_of_nine))]

        # 第一行表示batch的第一个, 产生的k^2 种情况的概率,列作为k^2情况的第几个
        # batch_prob.shape = (32,9)
        batch_prob = tf.concat(a, axis=1)

        # 取出topk,top_k_ids代表在每一个行中,最大的三种情况(0-8)
        top_k_probs, top_k_ids = tf.nn.top_k(batch_prob, k=params['beam_size'])

        # 记录新的三个产生的hyp
        my_tokens = [[] for _ in range(params['beam_size'])]
        my_prob = [[] for _ in range(params['beam_size'])]

        # 把每一句话的topk的token和log_probs取出来, 重新组合为一个新的Hypothesis
        for idx, each_sen in enumerate(top_k_ids):

            for number in range(params['beam_size']):
                my_tokens[number].append(list(hyp_of_nine[each_sen[number]].tokens[idx, :].numpy()))
                my_prob[number].append(list(hyp_of_nine[each_sen[number]].log_probs[idx, :].numpy()))

        # 生成一个新的结果集
        initial_hyp = [Hypothesis(tokens=my_tokens[i],
                                  log_probs=my_prob[i],
                                  hidden=dec_hidden,
                                  attn_dists=attention_weights) for i in range(params['beam_size'])]

        steps += 1

    final_hyp = sorted(initial_hyp, key=lambda x: x.avg_log_prob, reverse=True)

    results = []
    for sen in final_hyp[0].tokens:
        if stop_index in sen:
            sen = sen[:sen.index(stop_index)]
        results.append(''.join([reverse_vocab[idx] for idx in sen]))
    if is_full:
        return results
    else:
        return results[:shape[0]]

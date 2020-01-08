import tensorflow as tf
from utils.word2vec_dataload import Vocab
from utils.config import *

def article_to_ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word_to_id(Vocab.UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract_to_ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word_to_id(Vocab.UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def output_to_words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id_to_word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. \
            This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds \
                     to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    elif len(inp) == max_len:
        target.append(stop_id)
    else:
        target.append(stop_id)  # end token
        inp.append(stop_id)  # end token
    # assert len(inp) == len(target)
    return inp, target


def get_enc_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    if len(inp) >= max_len:  # truncate
        inp = inp[:max_len]
    else:
        inp.append(stop_id)  # end token
    # assert len(inp) == len(target)
    return inp


def example_generator(params, vocab, max_enc_len, max_dec_len, mode):
    if mode == "train":
        dataset_1 = tf.data.TextLineDataset(TRAIN_SEGMENT_X_DIALOGUE)
        dataset_2 = tf.data.TextLineDataset(TRAIN_SEGMENT_Y_DIALOGUE)
        train_dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
        train_dataset = train_dataset.shuffle(10, reshuffle_each_iteration=True)   # why use repeat ?

        for raw_record in train_dataset:

            article = raw_record[0].numpy().decode("utf-8")

            start_decoding = vocab.word_to_id(Vocab.START_DECODING)
            stop_decoding = vocab.word_to_id(Vocab.STOP_DECODING)

            article_words = article.split()[:max_enc_len]
            enc_len = len(article_words)

            # 添加mark标记
            sample_encoder_pad_mask = [1 for _ in range(enc_len)]

            enc_input = [vocab.word_to_id(w) for w in article_words]
            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

            # add start and stop flag
            enc_input = get_enc_inp_targ_seqs(enc_input, max_enc_len, start_decoding, stop_decoding)
            enc_input_extend_vocab = get_enc_inp_targ_seqs(enc_input_extend_vocab, max_enc_len, start_decoding,
                                                           stop_decoding)

            abstract = raw_record[1].numpy().decode("utf-8")
            abstract_words = abstract.split()
            abs_ids = [vocab.word_to_id(w) for w in abstract_words]

            dec_input, target = get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)

            if params['pointer_gen']:
                abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs)
                _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)

            dec_len = len(dec_input)

            # 添加mark标记
            sample_decoder_pad_mask = [1 for _ in range(dec_len)]

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": dec_input,
                "target": target,
                "dec_len": dec_len,
                "article": article,
                "abstract": abstract,
                "abstract_sents": abstract,
                "decoder_pad_mask": sample_decoder_pad_mask,
                "encoder_pad_mask": sample_encoder_pad_mask
            }
            yield output
    else:
        train_dataset = tf.data.TextLineDataset(TEST_SEGMENT_X_DIALOGUE)
        for raw_record in train_dataset:
            article = raw_record.numpy().decode("utf-8")
            article_words = article.split()[:max_enc_len]
            enc_len = len(article_words)

            enc_input = [vocab.word_to_id(w) for w in article_words]
            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

            # 添加mark标记
            sample_encoder_pad_mask = [1 for _ in range(enc_len)]

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": [],
                "target": [],
                "dec_len": params['max_dec_len'],
                "article": article,
                "abstract": '',
                "abstract_sents": '',
                "decoder_pad_mask": [],
                "encoder_pad_mask": sample_encoder_pad_mask
            }
            yield output


def batch_generator(generator, params, vocab, max_enc_len, max_dec_len, batch_size, mode):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(params, vocab, max_enc_len, max_dec_len, mode),
        output_types={
            "enc_len": tf.int32,
            "enc_input": tf.int32,
            "enc_input_extend_vocab": tf.int32,
            "article_oovs": tf.string,
            "dec_input": tf.int32,
            "target": tf.int32,
            "dec_len": tf.int32,
            "article": tf.string,
            "abstract": tf.string,
            "abstract_sents": tf.string,
            "decoder_pad_mask": tf.int32,
            "encoder_pad_mask": tf.int32,
        },
        output_shapes={
            "enc_len": [],
            "enc_input": [None],
            "enc_input_extend_vocab": [None],
            "article_oovs": [None],
            "dec_input": [None],
            "target": [None],
            "dec_len": [],
            "article": [],
            "abstract": [],
            "abstract_sents": [],
            "decoder_pad_mask": [None],
            "encoder_pad_mask": [None]
        })

    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=({"enc_len": [],
                                                   "enc_input": [None],
                                                   "enc_input_extend_vocab": [None],
                                                   "article_oovs": [None],
                                                   "dec_input": [max_dec_len],
                                                   "target": [max_dec_len],
                                                   "dec_len": [],
                                                   "article": [],
                                                   "abstract": [],
                                                   "abstract_sents": [],
                                                   "decoder_pad_mask": [max_dec_len],
                                                   "encoder_pad_mask": [None]
                                                   }),
                                   padding_values={"enc_len": -1,
                                                   "enc_input": vocab.word2id[Vocab.PAD_TOKEN],
                                                   "enc_input_extend_vocab": vocab.word2id[Vocab.PAD_TOKEN],
                                                   "article_oovs": b'',
                                                   "dec_input": vocab.word2id[Vocab.PAD_TOKEN],
                                                   "target": vocab.word2id[Vocab.PAD_TOKEN],
                                                   "dec_len": -1,
                                                   "article": b"",
                                                   "abstract": b"",
                                                   "abstract_sents": b'',
                                                   "decoder_pad_mask": 0,
                                                   "encoder_pad_mask": 0
                                                   },
                                   drop_remainder=True)

    def update(entry):
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 "max_oov_len": tf.shape(entry["article_oovs"])[1],
                 "encoder_pad_mask": entry["encoder_pad_mask"]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"],
                 "decoder_pad_mask": entry["decoder_pad_mask"]})

    dataset = dataset.map(update)
    return dataset


def batcher(vocab, params):
    dataset = batch_generator(example_generator,
                              params,
                              vocab,
                              params["max_enc_len"],
                              params["max_dec_len"],
                              params["batch_size"],
                              params["mode"])
    return dataset


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 获取参数
    params = get_params()
    params['mode'] = 'train'
    # vocab 对象
    vocab = Vocab(vocab_path)

    b = batcher(vocab, params)
    for i ,j in b:
        import pdb

        pdb.set_trace()
        print(i,j)

    print(next(iter(b)))

from utils.config import *
import numpy as np


class Vocab:
    PAD_TOKEN = 'PAD'
    UNKNOWN_TOKEN = 'UNK'
    START_DECODING = 'START'
    STOP_DECODING = 'STOP'

    def __init__(self, vocab_max_size=None):
        """
        Vocab 对象,vocab基本操作封装
        :param vocab_file: Vocab 存储路径
        :param vocab_max_size: 最大字典数量
        """
        self.word2id, self.id2word = self.load_vocab(vocab_max_size)
        self.count = len(self.word2id)

    @staticmethod
    def load_vocab(vocab_max_size=None):
        """
        读取字典
        :param file_path: 文件路径
        :return: 返回读取后的字典
        """
        vocab = {}
        reverse_vocab = {}
        for line in open(SAVE_VOCAB_PATH, "r", encoding='utf-8').readlines():
            word, index = line.strip().split("\t")
            index = int(index)
            # 如果vocab 超过了指定大小
            # 跳出循环 截断
            if vocab_max_size and index > vocab_max_size:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    vocab_max_size, index))
                break
            vocab[word] = index
            reverse_vocab[index] = word
        return vocab, reverse_vocab

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count

# def load_vocab():
#     vocab = {}
#     reverse_vocab = {}
#     with open(SAVE_VOCAB_PATH, 'r', encoding='utf-8') as f:
#         for data in f.readlines():
#             word, index = data.strip().split('\t')
#             vocab[word] = int(index)
#             reverse_vocab[int(index)] = word
#     return vocab, reverse_vocab


def load_embedding():
    return np.loadtxt(EMBEDDING_MATRIX_PATH)


def load_data():
    train_X = np.loadtxt(train_x_path).astype(int)
    train_Y = np.loadtxt(train_y_path).astype(int)
    TEST_X = np.loadtxt(test_x_path).astype(int)
    return train_X, train_Y, TEST_X

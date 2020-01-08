from utils.config import *
import pandas as pd
from utils.multi_process import parallelize
import jieba
import re
from gensim.models.word2vec import Word2Vec, LineSentence
import numpy as np
from utils.file_opeate import *


def build_dataset(train_path, test_path):
    """
    :param train_path: 训练集路径
    :param test_path: 测试集路径
    :return: 返回由词典对应索引的train_X, train_Y, test_X
             例如,train_x = [20,30,24,37,...,87]...
                train_X 对应的是dialogue question 的组合句子的序号
                train_y 对应的是训练集中的 report 句子的序号

                test_x 对应测试集中dialogue和question组合句子的序号

    处理思路如下:
        1, 导入数据
        2, 清除没有dialogue question report任何之一的信息
        3, 多进程处理dialogue question report中的内容.
            3.1 无用词清理
            3.2 切词, 为了清理停用词做准备
            3.3 停用词清理
        4, dialogue question report 合并一下, 构成训练集合, 并保存
        5, 第一次词向量训练,
        6, 对训练测试集统一输入输出长度: start+句子(unk)(pad)+end
        7, 第二次训练在第一次基础之上继续训练
        8, 建立词表, 把输入句子[我

    """
    # 源数据导入
    train_source = pd.read_csv(train_path, encoding='utf-8')
    test_source = pd.read_csv(test_path, encoding='utf-8')
    print('train data size {},test data size {}'.format(len(train_source), len(test_source)))

    # 生成专有名词词汇表
    train_source['Model'].to_csv(USER_DICT_PATH, index=False, header=False)
    jieba.load_userdict(USER_DICT_PATH)

    # drop null值
    train_source.dropna(subset=['Dialogue', 'Question', 'Report'], inplace=True)
    test_source.dropna(subset=['Dialogue', 'Question'], inplace=True)

    # 多进程数据处理, 清理无用词
    train_source = parallelize(train_source, source_process)
    test_source = parallelize(test_source, source_process)

    # 保存一下分好词的训练集
    train_source.to_csv(TRAIN_SEGMENT_PATH)
    test_source.to_csv(TEST_SEGMENT_PATH)

    # 为了word2vec词向量训练, 生成一句一句的切好词的文本
    train_source['Merge'] = train_source[['Dialogue', 'Question', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_source['Merge'] = test_source[['Dialogue', 'Question']].apply(lambda x: ' '.join(x), axis=1)

    # save
    train_source['X'] = train_source[['Dialogue', 'Question']].apply(lambda x: ' '.join(x), axis=1)
    train_source['X'].to_csv(TRAIN_SEGMENT_X_DIALOGUE, header=None, index=None)
    train_source['Report'].to_csv(TRAIN_SEGMENT_Y_DIALOGUE, header=None, index=None)
    test_source['Merge'].to_csv(TEST_SEGMENT_X_DIALOGUE, header=None, index=None)
    train_source.drop(['X'], axis=1)

    # 合并一下两个merge, 放如一列就行
    merge_seg_data = pd.concat([train_source['Merge'], test_source['Merge']])
    train_source = train_source.drop(['Merge'], axis=1)
    test_source = test_source.drop(['Merge'], axis=1)

    # 保存一下合并的分好词的句子文件,用于word2vec的训练
    merge_seg_data.to_csv(MERGE_SEGMENT_PATH, header=None, index=None)

    # 对词向量进行第一次训练
    print('start build w2v model')
    model = Word2Vec(LineSentence(MERGE_SEGMENT_PATH), workers=8, negative=5, min_count=5, size=300, window=3, iter=WV_EPOCH)
    vocab = model.wv.vocab

    # 统一训练的输入长度
    train_source['X'] = train_source[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_source['X'] = test_source[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    train_x_len = get_max_len(train_source['X'])
    test_x_len = get_max_len(test_source['X'])
    total_x_len = max(train_x_len, test_x_len)

    # 对长度不齐的整齐, 不足的用pad填充
    train_source['X'] = train_source['X'].apply(lambda x: padding_sentence(x, total_x_len, vocab))
    test_source['X'] = test_source['X'].apply(lambda x:padding_sentence(x, total_x_len, vocab))

    # 对输出也做同样的操作
    train_source['Y'] = train_source[['Report']]
    train_y_len = get_max_len(train_source['Y'])
    train_source['Y'] = train_source['Y'].apply(lambda x: padding_sentence(x, train_y_len, vocab))

    # 保存一下处理好长度的数据
    train_source['Y'].to_csv(TRAIN_PAD_Y_PATH, header=None, index=None)
    train_source['X'].to_csv(TRAIN_PAD_X_PATH, header=None, index=None)
    test_source['X'].to_csv(TEST_PAD_X_PATH, header=None, index=None)

    # 修改好了训练集合,在原来的基础再一次训练
    print('start retrain w2v model')
    model.build_vocab(LineSentence(TRAIN_PAD_X_PATH), update=True)
    model.train(LineSentence(TRAIN_PAD_X_PATH), epochs=WV_EPOCH, total_examples=model.corpus_count)

    model.build_vocab(LineSentence(TRAIN_PAD_Y_PATH), update=True)
    model.train(LineSentence(TRAIN_PAD_Y_PATH), epochs=WV_EPOCH, total_examples=model.corpus_count)

    model.build_vocab(LineSentence(TEST_PAD_X_PATH), update=True)
    model.train(LineSentence(TEST_PAD_X_PATH), epochs=WV_EPOCH, total_examples=model.corpus_count)
    print('finish retrain w2v model')

    # 保存词向量模型
    model.save(WV_MODEL_PATH)

    # 保存词向量
    np.savetxt(EMBEDDING_MATRIX_PATH, model.wv.vectors, fmt='%0.8f')

    # 建立字典
    vocab = {word: index for index, word in enumerate(model.wv.index2word)}

    save_dict(SAVE_VOCAB_PATH, vocab)

    # 把词转换为id
    train_x_ids = train_source['X'].apply(lambda x: transform(x, vocab))
    train_y_ids = train_source['Y'].apply(lambda x: transform(x, vocab))
    test_x_ids = test_source['X'].apply(lambda x: transform(x, vocab))

    train_X = np.array(train_x_ids.tolist())
    train_Y = np.array(train_y_ids.tolist())
    test_X = np.array(test_x_ids.tolist())

    np.savetxt(train_x_path, train_X, fmt='%0.8f')
    np.savetxt(train_y_path, train_Y, fmt='%0.8f')
    np.savetxt(test_x_path, test_X, fmt='%0.8f')

    return train_X, train_Y, test_X



def transform(x, vocab):
    '''
    把输入转换为对应词典的下标
    :param x:
    :param vocab:
    :return:
    '''
    return [vocab[word] for word in x.split()]



def padding_sentence(sentence, max_len, vocab):
    '''
    把句子长度统一为max_len, 添加 START PAD UNK END
    :param sentence: 句子
    :param max_len:  统一后的最长长度
    :param vocab:    词向量的词典
    :return:          经过修改的,长度统一的句子
    '''
    # 按照空格先截取
    words = sentence.strip().split()
    # 先截取
    words_cut = words[:max_len]
    # 判断是不是都在词典中,不在的用unk填充
    words_unk = [word if word in vocab else 'UNK' for word in words_cut]
    # 不到的长度的用pad填充
    if len(words_unk) < max_len:
        words_unk += ['PAD'] * (max_len - len(words_unk))

    return 'START ' + ' '.join(words_unk) + ' END'


def get_max_len(data):
    '''
    对一列,例如train_source['Question']
    :param data: 例如rain_source['Question']
    :return: 覆盖大约95%的句子分割长度
    '''
    lens = data.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(lens) + 2 * np.std(lens))

def source_process(df):
    """
    对 dataframe中数据进行处理 'Dialogue', 'Question' 和'Report',
    每一行分句处理

    :param df: 需要处理的dataframe
    :return: 处理完的dataframe
    """
    # 　取出需要的列，再做操作
    for column in ['Dialogue', 'Question']:
        print(column)
        df[column] = df[column].apply(lambda x: sentence_process(x))

    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(lambda x: sentence_process(x))

    return df


def sentence_process(sen):
    """
    对句子进行处理, 以字符串形式进行无用词处理, 以列表形式对停用词处理, 再组合为句子
    :param sen: 需要处理的句子
    :return: 处理后的句子
    """
    # 处理无用词
    sen = clean_sentence(sen)
    # 切词
    sen = jieba.cut(sen)
    # 过滤停用词
    stop_word = read_stop_words(STOP_WORD_PATH)
    data = [i for i in sen if i not in stop_word]
    return ' '.join(data)


def clean_sentence(data):
    """
    把句子中无用词替换为空
    :param data:
    :return:
    """

    return re.sub(r'[\s+\-\|\!\/\[\]\{\}_,$%^*(+\"\')]+|[:：+——()?【】“”、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
                          '', data) if isinstance(data, str) else ''


def read_stop_words(stop_word_path):
    """
    读取停用词表
    :param stop_word_path: 停用词表的路径
    :return: ->list
    """
    with open(stop_word_path, encoding='utf-8') as f:
        return [word.strip() for word in f.readlines()]


def load_train_data(max_enc_len, max_out_len):
    """
    :return: indexs of train data
    """
    train_x = np.loadtxt(train_x_path).astype(int)
    train_y = np.loadtxt(train_y_path).astype(int)
    return train_x[:, 0:max_enc_len], train_y[:, 0:max_out_len]


def load_test_data(max_enc_len):
    """
    :return: indexs of train data
    """
    test_x = np.loadtxt(test_x_path).astype(int)
    return test_x[:, 0:max_enc_len]


if __name__ == '__main__':
    build_dataset(TRAIN_PATH, TEST_PATH)

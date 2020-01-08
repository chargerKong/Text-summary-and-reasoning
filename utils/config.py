import os
import pathlib

# 获取根目录
root = pathlib.Path(__file__).parent.parent

# 训练集目录
TRAIN_PATH = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')

# 测试集目录
TEST_PATH = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')

# 停用词表
STOP_WORD_PATH = os.path.join(root, 'data', 'stopwords/哈工大停用词表.txt')

# 专有名词词汇表
USER_DICT_PATH = os.path.join(root, 'data', 'userdict.csv')

# 数据保存路径
# 保存训练数据的路径
TRAIN_SEGMENT_PATH = os.path.join(root, 'data', 'segment_x.csv')

# 保存测试数据的路径
TEST_SEGMENT_PATH = os.path.join(root, 'data', 'segment_y.csv')

# train_segment_x
TRAIN_SEGMENT_X_DIALOGUE = os.path.join(root, 'data', 'train_segment_x_dialogue.csv')
TRAIN_SEGMENT_Y_DIALOGUE = os.path.join(root, 'data', 'train_segment_y_dialogue.csv')
TEST_SEGMENT_X_DIALOGUE = os.path.join(root, 'data', 'test_segment_x_dialogue.csv')

# 保存训练词向量的分好句子的文件路径
MERGE_SEGMENT_PATH = os.path.join(root, 'data', 'merge_segment.csv')

# 词向量训练轮数
WV_EPOCH = 15

# 有了PAD之后的句子保存路径
TRAIN_PAD_Y_PATH = os.path.join(root, 'data', 'train_y_pad.csv')
TRAIN_PAD_X_PATH = os.path.join(root, 'data', 'train_x_pad.csv')
TEST_PAD_X_PATH = os.path.join(root, 'data', 'test_x_pad.csv')

# 词向量模型保存路径
WV_MODEL_PATH = os.path.join(root, 'data', 'word2vec', 'word2vec.model')

# 词向量矩阵保存路径
EMBEDDING_MATRIX_PATH = os.path.join(root, 'data', 'word2vec', 'embedding_matrix')


# 字典保存路径
SAVE_VOCAB_PATH = os.path.join(root, 'data', 'word2vec', 'vocab.txt')
SAVE_REVERSE_PATH = os.path.join(root, 'data', 'word2vec', 'reverse_vocab.txt')

#
train_x_path = os.path.join(root, 'data', 'train_x_ids')
train_y_path = os.path.join(root, 'data', 'train_y_ids')
test_x_path = os.path.join(root, 'data', 'test_x_ids')

# checkpoint path
checkpoint_path = os.path.join(root, 'data', 'checkpoint')

save_result_dir = os.path.join(root, 'result')

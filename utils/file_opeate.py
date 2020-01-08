import time
import os
from utils.config import save_result_dir


def save_dict(path, data):
    with open(path, 'w') as f:
        for i, j in data.items():
            f.write('{}\t{}\n'.format(i, j))


def get_result_filename(batch_size, epochs, max_length_inp, embedding_dim, commit=''):
    """
    获取时间
    :return:
    """
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    filename = now_time + '_batch_size_{}_epochs_{}_max_length_inp_{}_embedding_dim_{}{}.csv'.format(batch_size, epochs,
                                                                                                   max_length_inp,
                                                                                                   embedding_dim,
                                                                                                   commit)
    result_save_path = os.path.join(save_result_dir, filename)
    return result_save_path


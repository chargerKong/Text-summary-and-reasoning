import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, embedding_dim, embedding_matrix, batch_sz, vocab_size):
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, query, values, pre_coverage=None):
        # query为上次的GRU隐藏层
        # values为编码器的编码结果enc_output
        # 在seq2seq模型中，St是后面的query向量，而编码过程的隐藏状态hi是values。
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 计算注意力权重值
        if pre_coverage is not None:
            score = self.V(tf.nn.tanh(
                self.W1(values) + self.W2(hidden_with_time_axis) + self.W3(pre_coverage)))
            attention_weights = tf.nn.softmax(score, axis=1)
            coverage = pre_coverage + attention_weights
        else:
            score = self.V(tf.nn.tanh(
                self.W1(values) + self.W2(hidden_with_time_axis)))
            attention_weights = tf.nn.softmax(score, axis=1)
            coverage = attention_weights

        # attention_weights shape == (batch_size, max_length, 1)

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, tf.squeeze(attention_weights, -1), coverage


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

        # 用于注意力
        #self.attention = BahdanauAttention(self.dec_units)

    def __call__(self, x, hidden, enc_output, context_vector):
        # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
        #context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）

        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        dec_inp = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 将合并后的向量传送到 GRU
        output, state = self.gru(dec_inp)

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        pre = self.fc(output)

        return dec_inp, pre, state


class Point(tf.keras.layers.Layer):
    def __init__(self):
        self.w_h = tf.keras.layers.Dense(1)
        self.w_s = tf.keras.layers.Dense(1)
        self.w_x = tf.keras.layers.Dense(1)

    def __call__(self, context_vector, dec_hidden, dec_inp):
        """
        dec_inp represents the variable that dec_input after concat with context_vectotr
        :param context_vector:
        :param dec_hidden:
        :param dec_inp:
        :return: (batch, prob) the prob determines hwo much percent of the vocabulary predict to use
        """

        return tf.nn.sigmoid(self.w_h(context_vector) + self.w_s(dec_hidden) + self.w_x(dec_inp))

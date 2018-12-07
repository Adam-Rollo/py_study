# coding:utf-8

import argparse
import sys
import os
import io
import importlib
import time
import numpy as np
import collections
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.legacy_seq2seq as seq2seq

importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
tf.reset_default_graph()  

BEGIN_CHAR = '^'
END_CHAR = '$'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 100
MIN_LENGTH = 10
max_words = 3000
epochs = 5
poetry_file = 'poetry.txt'
save_dir = 'log'


class Data:
    def __init__(self):
        # 随机样本梯度下降，默认64个每次
        self.batch_size = 64
        self.poetry_file = poetry_file
        self.load()
        self.create_batches()

    def load(self):
        def handle(line):
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            return BEGIN_CHAR + line + END_CHAR

        # 从poetry里面一行一行的取出，去掉首尾和中间空格，并分割诗名和诗句
        self.poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                        io.open(self.poetry_file, encoding='utf-8')]
        # 十个字以下的诗直接不要了。100个字以上的，取100个字以内的一个句号作为分割，后面的句子也舍去。
        self.poetrys = [handle(line) for line in self.poetrys if len(line) > MIN_LENGTH]
        # 把每个字按照顺序放到words里面
        words = []
        for poetry in self.poetrys:
            words += [word for word in poetry]
        # class Counter(builtins.dict)
        # Dict subclass for counting hashable items.  Sometimes called a bag or multiset.  
        # Elements are stored as dictionary keys and their counts are stored as dictionary values.
        counter = collections.Counter(words)
        # dict_items([('b', 4), ('c', 3), ('a', 5), ('e', 1), ('d', 2)])
        # 按照字频降序排序
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
        # words = ['a','b','c','d','e'] _ = [5,4,3,2,1]
        words, _ = zip(*count_pairs)

        # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
        # 只取3000词:(1,2,3) + ('a',) = (1, 2, 3, 'a')
        words_size = min(max_words, len(words))
        self.words = words[:words_size] + (UNKNOWN_CHAR,)
        self.words_size = len(self.words)

        # 自然数ID和字互相映射
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        # 两个匿名方法
        self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num)
        
        # 把诗句按照长度排序
        # ['^绵绢，割两耳，只有面。$', '^桑条韦也，女时韦也乐。$', ....
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        display(self.poetrys)
        # [[2, 1227, 3000, 0, 2255, 252, 1002, 0, 283, 13, 536, 1, 3], [2, 746, 487, 2201, 906, 0, 334, 22, 2201, 906, 270, 1, 3],...
        # 其中3000是unknown char，所以有些映射到3000上
        self.poetrys_vector = [list(map(self.char2id, poetry)) for poetry in self.poetrys]
        
    def create_batches(self):
        # 分多少批进行训练，用总样本数整除每次训练的样本数
        self.n_size = len(self.poetrys_vector) // self.batch_size
        self.poetrys_vector = self.poetrys_vector[:self.n_size * self.batch_size]
        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            # 按顺序分组
            batches = self.poetrys_vector[i * self.batch_size: (i + 1) * self.batch_size]
            # 找到最长的一句
            length = max(map(len, batches))
            # 一行一行处理哦，将长度用星号补充到和最长的那个诗一样长
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]): length] = [self.unknow_char] * r
            # 把整个batches作为input，同样也作为output
            xdata = np.array(batches)
            ydata = np.copy(xdata)
            # 保留y的最后一列，并把x的所有列替换自己除了最后一列的所有列，相当于列+1
            ydata[:, :-1] = xdata[:, 1:]
            # 加入batches中
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)


class Model:
    def __init__(self, data, model='lstm', infer=False):
        # 每层RNN有128个神经元
        self.rnn_size = 128
        # 一共有2个RNN层
        self.n_layers = 2

        # 是否是推断模式，如果是推断模式，batch_size是64
        if infer:
            self.batch_size = 1
        else:
            self.batch_size = data.batch_size

        # 三种RNN网络
        if model == 'rnn':
            cell_rnn = rnn.BasicRNNCell
        elif model == 'gru':
            cell_rnn = rnn.GRUCell
        elif model == 'lstm':
            cell_rnn = rnn.BasicLSTMCell

         # 创建一个RNN层，可能是普通RNN，GRU或者LSTM
        cell = cell_rnn(self.rnn_size, state_is_tuple=False)
        #  复制上一个RNN，并且变成一个组合RNN
        self.cell = rnn.MultiRNNCell([cell] * self.n_layers, state_is_tuple=False)
        # 拥有64行N列的训练集和结果集
        self.x_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        # 初始化隐藏状态值，且全部设置为0
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        # 设置一个变量空间
        with tf.variable_scope('rnnlm'):
            # RNN层（隐藏层）到输出层的W是一个二维矩阵，Bias是一个一维向量
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, data.words_size])
            softmax_b = tf.get_variable("softmax_b", [data.words_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    "embedding", [data.words_size, self.rnn_size])
                # embedding_lookup is used to perform parallel lookups on the list of tensors in `params`. 
                # embedding将变量表现成了one-hot形式，而input_embedding = tf.nn.embedding_lookup(embedding, input_ids)就是把input_ids中给出的tensor表现成embedding中的形式。
                self.embedding = embedding
                inputs = tf.nn.embedding_lookup(embedding, self.x_tf)

        self.inputs = inputs       
                
        # 这里封装了输出层到隐藏层的线性运算，以及RNN内部的循环运算，以及LSTM的忘记，输入和输出
        outputs, final_state = tf.nn.dynamic_rnn(
            self.cell, inputs, initial_state=self.initial_state, scope='rnnlm')

        # output的shape是3000x128，softmax_w是128x3000，代表着3000个输入中每个输入对应的3000个输出的值
        self.output = tf.reshape(outputs, [-1, self.rnn_size])
        self.logits = tf.matmul(self.output, softmax_w) + softmax_b
        # 算3000个相关字中最高概率是哪个输出的字
        self.probs = tf.nn.softmax(self.logits)
        self.final_state = final_state
        pred = tf.reshape(self.y_tf, [-1])
        # seq2seq，也即是many to many，并且输入输出不等长不同步
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [pred],
                                                [tf.ones_like(pred, dtype=tf.float32)],)

        self.cost = tf.reduce_mean(loss)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        # 可以同时训练出softmax_w，softmax_b，以及输入层到RNN层的Weight和Bias，以及门各种门的各种参数
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


def train(data, model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        
        # display(help(tf.train.latest_checkpoint))
        model_file = tf.train.latest_checkpoint(save_dir)
        if model_file is not None:
            saver.restore(sess, model_file)

        n = 0
        for epoch in range(epochs):
            # 让训练速度越来越慢
            sess.run(tf.assign(model.learning_rate, 0.002 * (0.97 ** epoch)))
            pointer = 0
            for batche in range(data.n_size):
                n += 1
                feed_dict = {model.x_tf: data.x_batches[pointer], model.y_tf: data.y_batches[pointer]}
                pointer += 1
                
                # 看一看葫芦里卖的什么药
                [x_tf,embedding,inputs, outputs, init_state,logits,final_state] = sess.run([model.x_tf,model.embedding,model.inputs, model.output, model.initial_state, model.logits, model.final_state], feed_dict= feed_dict)
                display("X", np.shape(x_tf))
                display(x_tf)
                display("Embedding", np.shape(embedding))
                display(embedding)
                display("Input", np.shape(inputs))
                display(inputs)
                display("Outputs", np.shape(outputs) )
                display(outputs)
                display("初始state", np.shape(init_state))
                display(init_state)
                display("逻辑", np.shape(logits))
                display(logits)
                display("最终状态", np.shape(final_state))
                display(final_state)
                
                break
                
                train_loss, _, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed_dict)
                sys.stdout.write('\r')
                info = "{}/{} (epoch {}) | train_loss {:.3f}" \
                    .format(epoch * data.n_size + batche,
                            epochs * data.n_size, epoch, train_loss)
                sys.stdout.write(info)
                sys.stdout.flush()
                # save
                if (epoch * data.n_size + batche) % 1000 == 0 \
                        or (epoch == epochs-1 and batche == data.n_size-1):
                    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                    # display(checkpoint_path)
                    saver.save(sess, checkpoint_path, global_step=n)
                    sys.stdout.write('\n')
                    print("model saved to {}".format(checkpoint_path))
            sys.stdout.write('\n')


def sample(data, model, head=u''):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sa = int(np.searchsorted(t, np.random.rand(1) * s))
        return data.id2char(sa)

    for word in head:
        if word not in data.words:
            return u'{} 不在字典中'.format(word)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(save_dir)
        # print(model_file)
        saver.restore(sess, model_file)

        if head:
            print('生成藏头诗 ---> ', head)
            poem = BEGIN_CHAR
            for head_word in head:
                poem += head_word
                x = np.array([list(map(data.char2id, poem))])
                state = sess.run(model.cell.zero_state(1, tf.float32))
                feed_dict = {model.x_tf: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = to_word(probs[-1])
                while word != u'，' and word != u'。':
                    poem += word
                    x = np.zeros((1, 1))
                    x[0, 0] = data.char2id(word)
                    [probs, state] = sess.run([model.probs, model.final_state],
                                              {model.x_tf: x, model.initial_state: state})
                    word = to_word(probs[-1])
                poem += word
            return poem[1:]
        else:
            poem = ''
            head = BEGIN_CHAR
            x = np.array([list(map(data.char2id, head))])
            state = sess.run(model.cell.zero_state(1, tf.float32))
            feed_dict = {model.x_tf: x, model.initial_state: state}
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
            word = to_word(probs[-1])
            while word != END_CHAR:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = data.char2id(word)
                [probs, state] = sess.run([model.probs, model.final_state],
                                          {model.x_tf: x, model.initial_state: state})
                word = to_word(probs[-1])
            return poem


def main():
    msg = """
            Usage:
            Training: 
                python poetry_gen.py --mode train
            Sampling:
                python poetry_gen.py --mode sample --head 明月别枝惊鹊
            """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sample',
                        help=u'usage: train or sample, sample is default')
#     parser.add_argument('--mode', type=str, default='train',
#                         help=u'usage: train or sample, sample is default')
    parser.add_argument('--head', type=str, default='',
                        help='生成藏头诗')

    args = parser.parse_args()

    if args.mode == 'sample':
        infer = True  # True
        data = Data()
        model = Model(data=data, infer=infer)
        print(sample(data, model, head=args.head))
    elif args.mode == 'train':
        infer = False
        data = Data()
        model = Model(data=data, infer=infer)
        print(train(data, model))
    else:
        print(msg)


if __name__ == '__main__':
    main()

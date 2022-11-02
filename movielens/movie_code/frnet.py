import os
import numpy as np
import tensorflow as tf
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def normalize(inputs, epsilon=1e-8):
    '''
    Applies layer normalization
    Args:
        inputs: A tensor with 2 or more dimensions
        epsilon: A floating number to prevent Zero Division
    Returns:
        A tensor with the same shape and data dtype
    '''
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

    return outputs


def single_attention(values, att_size, embedding_size):

    Q = tf.layers.dense(values, att_size, activation=tf.nn.relu)
    K = tf.layers.dense(values, att_size, activation=tf.nn.relu)
    V = tf.layers.dense(values, att_size, activation=tf.nn.relu)

    # Multiplication
    weights = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

    # Activation
    weights = tf.nn.softmax(weights)

    # Weighted sum
    context = tf.matmul(weights, V)

    # Projection
    context = tf.layers.dense(context, embedding_size, activation=tf.nn.relu)

    return context


def PReLU(_x, i):
    with tf.variable_scope("PReLU", reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('p_%d' % i, _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - tf.abs(_x)) * 0.5
    return pos + neg


class FRNet():
    def __init__(self, args, feature_size, run_cnt):

        self.feature_size = feature_size  # denote as n, dimension of concatenated features
        self.field_size = args.field_size  # denote as M, number of total feature fields
        self.embedding_size = args.embedding_size  # denote as d, size of the feature embedding
        self.blocks = args.blocks  # number of the blocks
        self.heads = args.heads  # number of the heads
        self.block_shape = args.block_shape
        self.output_size = args.block_shape[-1]
        self.has_residual = args.has_residual
        self.deep_layers = args.deep_layers  # whether to joint train with deep networks as described in paper

        # super parameter follow the origin paper
        self.att_size = 64
        self.bit_layer = 1
        self.cie_mlp_layer = 256
        self.weight_type = 'bit'

        self.batch_norm = args.batch_norm
        self.batch_norm_decay = args.batch_norm_decay
        self.drop_keep_prob = args.dropout_keep_prob
        self.l2_reg = args.l2_reg
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.optimizer_type = args.optimizer_type

        self.save_path = args.save_path + str(run_cnt) + '/'
        self.is_save = args.is_save
        if (args.is_save == True and os.path.exists(self.save_path) == False):
            os.makedirs(self.save_path)

        self.verbose = args.verbose
        self.random_seed = args.random_seed
        self.loss_type = args.loss_type
        self.eval_metric = roc_auc_score
        self.best_loss = 1.0
        self.greater_is_better = args.greater_is_better
        self.train_result, self.valid_result = [], []
        self.train_loss, self.valid_loss = [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            # placeholder for single-value field.
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name="feat_index")  # None * M-1
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                             name="feat_value")  # None * M-1

            # placeholder for multi-value field. (movielens dataset genre field)
            self.genre_index = tf.placeholder(tf.int32, shape=[None, None],
                                              name="genre_index")  # None * 6
            self.genre_value = tf.placeholder(tf.float32, shape=[None, None],
                                              name="genre_value")  # None * 6

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1

            # In our implementation, the shape of dropout_keep_prob is [3], used in 3 different places.
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_prob")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                     self.feat_index)  # None * M-1 * d
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size - 1, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)  # None * M-1 * d

            # for multi-value field
            self.embeddings_m = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                       self.genre_index)  # None * 6 * d
            genre_value = tf.reshape(self.genre_value, shape=[-1, 6, 1])
            self.embeddings_m = tf.multiply(self.embeddings_m, genre_value)
            self.embeddings_m = tf.reduce_sum(self.embeddings_m, axis=1)  # None * d
            self.embeddings_m = tf.div(self.embeddings_m,
                                       tf.reduce_sum(self.genre_value, axis=1, keep_dims=True))  # None * d

            # concatenate single-value field with multi-value field
            self.embeddings = tf.concat([self.embeddings, tf.expand_dims(self.embeddings_m, 1)], 1)  # None * M * d

            # IEU_G computes complementary features.
            com_feature = self.IEU()
            weight_matrix = tf.nn.sigmoid(self.IEU(weight_type=self.weight_type))
            # CSGate
            x_out = self.embeddings * weight_matrix + com_feature * (tf.ones_like(weight_matrix) - weight_matrix)

            # linear loss
            self.flat = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
            linear_loss = tf.add(tf.matmul(self.flat, self.weights["prediction_dense"]),
                                 self.weights["prediction_bias_dense"], name='logits_dense')

            # FM loss
            square_of_sum = tf.reduce_sum(x_out, axis=1) ** 2  # B，embed_dim
            sum_of_square = tf.reduce_sum(x_out ** 2, axis=1)  # B，embed_dim
            # square of sum - sum of square
            ix = square_of_sum - sum_of_square  # B, embed_dim
            ix = tf.reduce_sum(ix, axis=1, keepdims=True)
            fm_loss = 0.5 * ix

            self.out = linear_loss + fm_loss

            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out, name='pred')
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.var1 = [v for v in tf.trainable_variables() if v.name != 'feature_bias:0']
            self.var2 = [tf.trainable_variables()[1]]  # self.var2 = [feature_bias]

            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        beta1=0.9, beta2=0.999, epsilon=1e-8). \
                    minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8). \
                    minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate). \
                    minimize(self.loss, global_step=self.global_step)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95). \
                    minimize(self.loss, global_step=self.global_step)

            # init
            self.saver = tf.train.Saver(max_to_keep=2)
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
            self.count_param()

    def count_param(self):
        k = (np.sum([np.prod(v.get_shape().as_list())
                     for v in tf.trainable_variables()]))

        print("total parameters :%d" % k)
        print("extra parameters : %d" % (k - self.feature_size * self.embedding_size))

    def _init_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()

        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size(n) * d

        glorot = 0.01
        weights["mlp_layer_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.field_size * self.embedding_size, self.cie_mlp_layer)), dtype=np.float32)
        weights["mlp_bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.cie_mlp_layer)), dtype=np.float32)  # 1 * layers[0]
        for i in range(1, self.bit_layer):
            weights["mlp_layer_%d" % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.cie_mlp_layer, self.cie_mlp_layer)), dtype=np.float32)  # layers[i-1] * layers[i]
            weights["mlp_bias_%d" % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.cie_mlp_layer)), dtype=np.float32)  # 1 * layer[i]

        weights["bit_projection"] = tf.Variable(np.random.normal(loc=0, scale=0.01, size=(self.cie_mlp_layer, self.embedding_size)), dtype=np.float32,name="bit_projection")
        weights["bit_projection_bias"] = tf.Variable(np.random.normal(), dtype=np.float32, name="bit_projection_bias")

        weights["prediction_dense"] = tf.Variable(np.random.normal(loc=0, scale=0.01, size=(self.field_size * self.embedding_size, 1)), dtype=np.float32, name="prediction_dense")
        weights["prediction_bias_dense"] = tf.Variable(np.random.normal(), dtype=np.float32, name="prediction_bias_dense")

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def get_batch(self, Xi, Xv, Xi_genre, Xv_genre, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], Xi_genre[start:end], Xv_genre[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c, d, e):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)

    def fit_on_batch(self, Xi, Xv, Xi_genre, Xv_genre, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.genre_index: Xi_genre,
                     self.genre_value: Xv_genre,
                     self.label: y,
                     self.dropout_keep_prob: self.drop_keep_prob,
                     self.train_phase: True}
        step, loss, opt = self.sess.run((self.global_step, self.loss, self.optimizer), feed_dict=feed_dict)
        return step, loss

    # Since the train data is very large, they can not be fit into the memory at the same time.
    # We separate the whole train data into several files and call "fit_once" for each file.
    def fit_once(self, Xi_train, Xv_train, Xi_train_genre, Xv_train_genre, y_train,
                 epoch, Xi_valid=None,
                 Xv_valid=None, Xi_valid_genre=None, Xv_valid_genre=None, y_valid=None,
                 early_stopping=False):

        has_valid = Xv_valid is not None
        last_step = 0
        t1 = time()
        self.shuffle_in_unison_scary(Xi_train, Xv_train, Xi_train_genre, Xv_train_genre, y_train)
        total_batch = int(len(y_train) / self.batch_size)
        for i in range(total_batch):
            Xi_batch, Xv_batch, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi_train, Xv_train,
                                                                                         Xi_train_genre, Xv_train_genre,
                                                                                         y_train, self.batch_size, i)
            step, loss = self.fit_on_batch(Xi_batch, Xv_batch, Xi_batch_genre, Xv_batch_genre, y_batch)
            print("step:%d-loss:%.4f" % (step, loss))
            last_step = step

        # evaluate training and validation datasets
        train_result, train_loss = self.evaluate(Xi_train, Xv_train, Xi_train_genre, Xv_train_genre, y_train)
        self.train_result.append(train_result)
        self.train_loss.append(train_loss)
        if has_valid:
            valid_result, valid_loss = self.evaluate(Xi_valid, Xv_valid, Xi_valid_genre, Xv_valid_genre, y_valid)
            self.valid_result.append(valid_result)
            self.valid_loss.append(valid_loss)
            if valid_loss < self.best_loss and self.is_save == True:
                old_loss = self.best_loss
                self.best_loss = valid_loss
                self.saver.save(self.sess, self.save_path + 'model.ckpt', global_step=last_step)
                print("[%d] model saved!. Valid loss is improved from %.4f to %.4f"
                      % (epoch, old_loss, self.best_loss))

        if self.verbose > 0 and ((epoch - 1) * 9) % self.verbose == 0:
            if has_valid:
                print("[%d] train-result=%.4f, train-logloss=%.4f, valid-result=%.4f, valid-logloss=%.4f [%.1f s]" % (
                epoch, train_result, train_loss, valid_result, valid_loss, time() - t1))
            else:
                print("[%d] train-result=%.4f [%.1f s]" \
                      % (epoch, train_result, time() - t1))
        if has_valid and early_stopping and self.training_termination(self.valid_loss):
            return False
        else:
            return True

    def training_termination(self, valid_result):
        if len(valid_result) > 4:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] < valid_result[-3] < valid_result[-4]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] > valid_result[-3] > valid_result[-4]:
                    return True
        return False

    def predict(self, Xi, Xv, Xi_genre, Xv_genre):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi, Xv, Xi_genre, Xv_genre,
                                                                                     dummy_y, self.batch_size,
                                                                                     batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.genre_index: Xi_batch_genre,
                         self.genre_value: Xv_batch_genre,
                         self.label: y_batch,
                         self.dropout_keep_prob: [1.0] * len(self.drop_keep_prob),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, Xi_batch_genre, Xv_batch_genre, y_batch = self.get_batch(Xi, Xv, Xi_genre, Xv_genre,
                                                                                         dummy_y, self.batch_size,
                                                                                         batch_index)

        return y_pred

    def evaluate(self, Xi, Xv, Xi_genre, Xv_genre, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv, Xi_genre, Xv_genre)
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
        return self.eval_metric(y, y_pred), log_loss(y, y_pred)

    def restore(self, save_path=None):
        if (save_path == None):
            save_path = self.save_path
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            if self.verbose > 0:
                print("restored from %s" % (save_path))

    def IEU(self, weight_type='bit'):
        # （1）self-attetnion unit
        x_vector = single_attention(self.embeddings, self.att_size, self.embedding_size)
        # (2) CIE unit
        mlp_input = tf.reshape(self.embeddings, [-1, self.field_size * self.embedding_size])
        for i in range(self.bit_layer):
            output = tf.add(tf.matmul(mlp_input, self.weights["mlp_layer_%d" % i]), self.weights["mlp_bias_%d" % i])
            output = normalize(output)
            # output = PReLU(output, i)
            output = tf.nn.relu(output)
            # output = tf.nn.dropout(output, 0.5)
            mlp_input = output

        x_bit = tf.add(tf.matmul(output, self.weights['bit_projection']), self.weights["bit_projection_bias"])
        x_bit = tf.expand_dims(x_bit, axis=1)
        x_bit = tf.nn.relu(x_bit)
        # （3）integration unit
        x_out = x_bit * x_vector

        if weight_type == "vector":
            # To compute vector-level importance in IEU_W
            x_out = tf.reduce_sum(x_out, axis=2, keepdims=True)
            # B * F *1
            return x_out
        return x_out

import os
import numpy as np
import tensorflow as tf
from time import time
from sklearn.metrics import roc_auc_score, log_loss
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def interaction_layer(values, field_size, lay_count, embedding_size):
    embedding = values
    res = [values]
    for i in range(lay_count):
        field_aware_interaction = interaction(embedding, field_size)

        vector_agg = tf.layers.dense(embedding, embedding_size, activation=tf.nn.relu, use_bias=False)
        vector_aware_interaction = interaction(vector_agg, field_size)
        embedding = tf.concat((field_aware_interaction, vector_aware_interaction), axis=-1)
        res.append(embedding)
    return embedding
    # return tf.concat(res, axis=-1)


def interaction(value, field_num):
    value_t = tf.pad(value, paddings=[[0, 0], [0, 1], [0, 0]], constant_values=1)
    value_t = tf.transpose(value_t, perm=[0, 2, 1])
    context_vec = tf.layers.dense(value_t, field_num, use_bias=False)
    context_vec = tf.transpose(context_vec, perm=[0, 2, 1])
    return value * context_vec


def attention_layer(value, embedding_size):
    k = int(value.shape[-1].value / embedding_size)
    split = tf.concat(tf.split(value, k, axis=-1), axis=0)
    attention = tf.layers.dense(split, 1, activation=tf.nn.relu)
    weighted_sum = attention * split
    weighted_sum = tf.concat(tf.split(weighted_sum, k, axis=0), axis=-1)
    return weighted_sum


class EFANet():
    def __init__(self, args, feature_size, cnt):

        self.feature_size = feature_size  # denote as n, dimension of concatenated features
        self.field_size = args.field_size  # denote as M, number of total feature fields
        self.embedding_size = args.embedding_size  # denote as d, size of the feature embedding
        self.blocks = args.blocks  # number of the blocks
        self.heads = args.heads  # number of the heads
        self.block_shape = args.block_shape
        self.has_residual = args.has_residual
        self.l2_reg = args.l2_reg
        self.deep_layers = args.deep_layers

        self.interaction_layer = 3
        self.full_connection_size = args.field_size * args.embedding_size * 4

        self.batch_norm = args.batch_norm
        self.batch_norm_decay = args.batch_norm_decay
        self.drop_keep_prob = args.dropout_keep_prob
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.optimizer_type = args.optimizer_type

        self.data_path = args.data_path

        self.save_path = args.save_path + str(cnt) + "/"
        self.is_save = args.is_save
        if args.is_save and os.path.exists(self.save_path) is not True:
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

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")

            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_prob")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            interaction_layer_output = interaction_layer(self.embeddings, self.field_size, self.interaction_layer, self.embedding_size)
            attention_output = attention_layer(interaction_layer_output, self.embedding_size)

            self.y_dense = tf.reshape(attention_output, shape=[-1, self.full_connection_size])
            for i in range(0, len(self.deep_layers)):
                self.y_dense = tf.add(tf.matmul(self.y_dense, self.weights["layer_%d" %i]), self.weights["bias_%d"%i])
                self.y_dense = tf.nn.relu(self.y_dense)
            self.out = tf.add(tf.matmul(self.y_dense, self.weights["prediction_dense"]), self.weights["prediction_bias_dense"], name='logits_dense')  # None * 1

            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out, name='pred')
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8). \
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

            self.saver = tf.train.Saver(max_to_keep=2)
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
            self.count_param()

    def count_param(self):
        k = (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

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
            name="feature_embeddings")

        if self.deep_layers is not None:
            num_layer = len(self.deep_layers)
            layer0_size = self.full_connection_size
            glorot = np.sqrt(2.0 / (layer0_size + self.deep_layers[0]))
            weights["layer_0"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(layer0_size, self.deep_layers[0])), dtype=np.float32)
            weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                            dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1] * layers[i]
                weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            glorot = np.sqrt(2.0 / (self.deep_layers[-1] + 1))
            weights["prediction_dense"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                dtype=np.float32, name="prediction_dense")
            weights["prediction_bias_dense"] = tf.Variable(
                np.random.normal(), dtype=np.float32, name="prediction_bias_dense")

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_prob: self.drop_keep_prob,
                     self.train_phase: True}
        step, loss, opt = self.sess.run((self.global_step, self.loss, self.optimizer), feed_dict=feed_dict)
        return step, loss

    def fit_once(self, Xi_train, Xv_train, y_train, epoch, file_count, Xi_valid=None, Xv_valid=None, y_valid=None, early_stopping=False):

        has_valid = Xv_valid is not None
        last_step = 0
        t1 = time()
        self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
        total_batch = int(len(y_train) / self.batch_size)
        for i in range(total_batch):
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
            step, loss = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
            # if epoch == 1:
            #     print("step:%d-loss:%.4f" % (step, loss))
            last_step = step

        # evaluate training and validation datasets
        train_result, train_loss = self.evaluate(Xi_train, Xv_train, y_train)
        self.train_result.append(train_result)
        self.train_loss.append(train_loss)
        if has_valid:
            valid_result, valid_loss = self.evaluate(Xi_valid, Xv_valid, y_valid)
            self.valid_result.append(valid_result)
            self.valid_loss.append(valid_loss)
            if valid_loss < self.best_loss and self.is_save == True:
                old_loss = self.best_loss
                self.best_loss = valid_loss
                self.saver.save(self.sess, self.save_path + 'model.ckpt', global_step=last_step)
                print("[%d-%d] model saved!. Valid loss is improved from %.4f to %.4f"
                      % (epoch, file_count, old_loss, self.best_loss))

        if self.verbose > 0 and ((epoch - 1) * 9 + file_count) % self.verbose == 0:
            if has_valid:
                print(
                    "[%d-%d] train-result=%.4f, train-logloss=%.4f, valid-result=%.4f, valid-logloss=%.4f [%.1f s]" % (
                        epoch, file_count, train_result, train_loss, valid_result, valid_loss, time() - t1))
            else:
                print("[%d-%d] train-result=%.4f [%.1f s]" \
                      % (epoch, file_count, train_result, time() - t1))
        if has_valid and early_stopping and self.training_termination(self.valid_loss):
            return False
        else:
            return True

    def training_termination(self, valid_result):
        if len(valid_result) > 3:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] < valid_result[-3]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] > valid_result[-3]:
                    return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_prob: [1.0] * len(self.drop_keep_prob),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
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
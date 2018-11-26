import os

import numpy as np
import tensorflow as tf

class DEEPER:

    def __init__(self,n_hidden=100, embed=None, batch_size=128):
        self.n_hidden = n_hidden
        self.embed = embed
        self.batch_size = batch_size
        with tf.variable_scope('deeper', reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self._create_network()
            self._create_losses()
            self._create_optimizers()
            self._create_metrics()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.best_val_acc = -float('inf')
        self.best_val_recon_loss = float('inf')



    def _build_lstm_layers(n_hidden, embed, batch_size, reuse=tf.AUTO_REUSE):
        """
        Create computation graph that returns output tensors of the recognition
        network: a tensor of means and a tensor of log standard deviations that
        define the factorized latent distribution q(z).
        """
        with tf.variable_scope('lstm', reuse=reuse):
            lstms = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            # Add dropout to the cell
            # drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
            # Stack up multiple LSTM layers, for deep learning
            #cell = tf.contrib.rnn.MultiRNNCell(drops)
            # Getting an initial state of all zeros
            initial_state = lstms.zero_state(batch_size, tf.float32)
            # generate prediction
            lstm_outputs, final_state = tf.nn.dynamic_rnn(lstms, embed, initial_state=initial_state)

            return (lstm_outputs, final_state)

    def _create_network(self):
        """
        Define the entire computation graph
        """
        self.x_input_similarities = tf.placeholder(
            tf.float32, shape=[None, self.similariy_shape], name='x_input')
        self.y_input = tf.placeholder(
            tf.float32, shape=[None, 2], name='y_input')
        self.learning_rate = tf.placeholder(
            tf.float32, name='learning_rate')

        # Build embedding layer

        # Build LSTM layer
        self.lstm_outputs, self.final_state = self._build_lstm_layers(self.n_hidden, self.embed,
                                                                      self.batch_size)
        print(self.lstm_outputs.shape)

        # Concatenate other features
        # TODO: take output of LSTM and contatenate with all other features to form vector z

        # Build Dense layers and get logits
        self.y_logits = self._create_classifier_network(self.z)

    def _create_classifier_network(self, z):
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            #dense = tf.layers.Dense(256, activation=tf.nn.relu)(z)
            #dropout = tf.layers.dropout(inputs=dense, rate=0.2)
            y_logits = tf.layers.Dense(self.num_classes, name='y_logits')(z)
        return y_logits

    def _create_losses(self):

        # Binary cross-entropy loss
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
         labels=self.y_input, logits=self.y_logits)
        self.cross_entropy_loss = tf.reduce_mean(
            cross_entropy_loss, name='cross_entropy_loss')

    def _create_optimizers(self):

        optimizer = tf.train.AdamOptimizer
        classifier_optimizer = optimizer(self.learning_rate)
        self.classifier_optimizer = classifier_optimizer.minimize(
            self.cross_entropy_loss,
            global_step=self.global_step)

    def _create_metrics(self):

        y_labels = tf.argmax(self.y_input, axis=1)
        self.accuracy, self.accuracy_update = tf.metrics.accuracy(
            predictions=self.y_pred_labels,
            labels=y_labels, name="accuracy")
        self.running_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope=tf.get_variable_scope().name)
        # print(self.running_vars)
        # Define initializer to initialize/reset running variables
        self.running_vars_initializer = tf.variables_initializer(
            var_list=self.running_vars)

    def _partial_fit_classifier(self, x_batch, y_batch, learning_rate, beta):
        """
        Train encoder and classifier networks based on minibatch
        of training data.

        Parameters
        ----------
        x_batch : array-like, shape = [batch_size, height, width, channels]
            A minibatch of input images.
        y_batch : array-like, shape = [batch_size, num_classes]
            A one-hot encoded matrix of training labels.
        """

        feed_dict = {
            self.x_input: x_batch,
            self.y_input: y_batch,
            self.learning_rate: learning_rate,
            self.beta: beta
        }
        _ = self.sess.run(self.classifier_optimizer, feed_dict=feed_dict)

        step = self.sess.run(self.global_step)

        print(
            'cross_entropy_loss:',
            self.sess.run(self.cross_entropy_loss, feed_dict=feed_dict),
            end='\t')
        print(
            'latent_loss:',
            self.sess.run(self.latent_loss, feed_dict=feed_dict))

    def fit_classifier(
        self, x, y, num_epochs=5, batch_size=256,
        learning_rate=1e-3, beta=1):
        """
        Train encoder and classifier networks.

        Parameters
        ----------
        x : array-like, shape = [num_samples, height, width, channels]
            A set of input images.
        y : array-like, shape = [num_samples, num_classes]
            A one-hot encoded matrix of training labels.
        """
        # Shuffle x and y
        num_samples = len(x)
        for epoch in range(num_epochs):
            random_indices = np.random.permutation(num_samples)
            x = x[random_indices]
            y = y[random_indices]

            # Split x and y into batches
            num_batches = num_samples // batch_size
            indices = [[k, k+batch_size] for k in range(0, num_samples, batch_size)]
            indices[-1][-1] = num_samples
            x_batches = [x[start:end] for start, end in indices]
            y_batches = [y[start:end] for start, end in indices]

            print('Training epoch {}...'.format(epoch))
            # Iteratively train the classifier
            for x_batch, y_batch in zip(x_batches, y_batches):
                self._partial_fit_classifier(
                    x_batch, y_batch, learning_rate, beta)

    def _calc_classifier_metrics(self, data_generator):
        """
        Calculate accuracy and loss metrics for input data,
        based on the existing network in current session.

        Parameters
        ----------
        data_generator : python generator
            a generator on either train or val dataset.

        Return
        ---------
        acc : scalar
            accuracy score over whole dataset.
        loss : scalar
            loss over whole dataset.
        """

        _ = self.sess.run(self.running_vars_initializer)
        loss_list = []

        while True:
            x_batch, y_batch = data_generator.next()
            y_batch_cat = tf.keras.utils.to_categorical(y_batch, num_classes=10)

            feed_dict = {
                self.x_input: x_batch,
                self.y_input: y_batch_cat
            }
            # Calulate accuracy
            self.sess.run(self.accuracy_update,
                                feed_dict=feed_dict)

            # Calculate loss
            batch_loss_realized = self.sess.run(self.cross_entropy_loss,
                                feed_dict=feed_dict)
            loss_list.append(batch_loss_realized)

            # print('Predicting batch {} of {}'.format(
            # 	data_generator.batch_index,
            # 	len(data_generator)))

            if data_generator.batch_index == 0:
                break
        acc = self.sess.run(self.accuracy)
        loss = np.mean(loss_list)

        return acc, loss

    def predict_proba(self, x):
        """
        Given a minibatch of input images, predict classes.

        Parameters
        ==========
        x : array-like, shape = [batch_size, height, width, channels]
            A minibatch of input images.

        Returns
        =======
        predictions : array, shape = [batch_size, num_classes]
            A matrix of <batch_size> predictive distributions.
        """

        feed_dict = {
            self.x_input: x,
        }
        predictions = self.sess.run(self.y_pred_denoised, feed_dict=feed_dict)
        return predictions

    def predict_label(self,y_proba):
        """
        Generate predicted labels given probabilities

        Parameters
        ==========
        y_proba : array-like, shape = [batch_size,num_classes]
            A matrix of <batch_size> probabilities.

        Returns
        =======
        prediction_labels : array, shape = [batch_size, num_classes]
            A matrix of <batch_size> predicted labels.
        """
        y_predict_label = tf.argmax(y_proba, axis=1)

        return y_predict_label


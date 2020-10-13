#encoding=utf-8

import sys
import os

import tensorflow as tf

def model_fn(features, labels, mode, params, config=None):
    # Do some checks.
    # corpus_size must be in params indicating the total num of feature keys.
    assert "corpus_size" in params
    assert "batch_size" in params
    assert "learning_rate" in params
    assert "embedding_size" in params
    assert "dropout_rate" in params
    assert "optimizer" in params
    assert "batch_norm" in params

    corpus_size = params["corpus_size"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    embedding_size = params["embedding_size"]
    # droupout rate: 1 is no-dropout, 0 is all-dropout
    dropout_rate = params["dropout_rate"]
    # True or False
    batch_norm = params["batch_norm"]
    # optimizer is "sgd", "adagrad", "adam"
    optimizer = params["optimizer"]
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.name_scope("embedding"):
        embedding_matrix = tf.Variable(tf.random_normal([corpus_size, embedding_size], 0.0, 0.01), name="embedding_matrix")
        bias_matrix = tf.Variable(tf.random_uniform([corpus_size, 1], 0.0, 0.0), name='feature_bias')
        feature_emb = tf.nn.embedding_lookup(embedding_matrix, features, name="feature_embedding_lookup")
        feature_bias = tf.nn.embedding_lookup(bias_matrix, features, name="feature_bias_lookup")
        summed_feature_emb = tf.reduce_sum(feature_emb, axis=1)

    with tf.name_scope("fm"):
        # get the element-multiplication
        summed_feature_emb_square = tf.square(summed_feature_emb)
        # _________ square_sum part _____________
        squared_feature_emb = tf.square(feature_emb)
        squared_sum_feature_emb = tf.reduce_sum(squared_feature_emb, axis=1)
        # ________ FM __________
        fm = 0.5 * tf.subtract(summed_feature_emb_square, squared_sum_feature_emb, name="fm")
        if batch_norm:
            bn_layer = tf.layers.BatchNormalization()
            fm = bn_layer(fm, training=is_training)
        elif dropout_rate < 1:
            dropout_layer = tf.layers.Dropout(rate=dropout_rate)
            fm = dropout_layer(fm, training=is_training)
        # _________out _________
        bilinear = tf.reduce_sum(fm, axis=1)  # None * 1
        bias = tf.reduce_sum(feature_bias, axis=1)
        y_pred = tf.add_n([bilinear, bias], name="y_pred")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'y_pred': y_pred,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    THRESHOLD = 0.5
    with tf.name_scope("loss"):
        y_expand = tf.reshape(labels, shape=[-1, 1])
        y_float = tf.to_float(y_expand)
        likelyhood = -(y_float * tf.log(y_pred) + (1.0 - y_float) * (tf.log(1.0 - y_pred)))
        loss = tf.reduce_mean(likelyhood, axis=0)

        # Metrics
        one_like = tf.ones_like(y_expand, dtype=tf.int32)
        zero_like = tf.zeros_like(y_expand, dtype=tf.int32)
        y_pred_int = tf.where(y_pred < THRESHOLD, x=zero_like, y=one_like)
        corrections = tf.equal(y_expand, y_pred_int)
        accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))

    logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                               "accuracy": accuracy}, every_n_iter=1)

    if mode == tf.estimator.ModeKeys.EVAL:
        tf.summary.scalar("test_acc", accuracy)
        acc = tf.metrics.accuracy(y_expand, y_pred)
        metrics = {'acc': acc}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[logging_hook])

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar("train_acc", accuracy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    fake_input = tf.constant(
        [
            [2, 5, 8, 1],
            [2, 6, 1, 3],
            [9, 0, 4, 7]
        ], shape=(3, 4))
    fake_label = tf.constant([1, 1, 0], shape=(1, 3))
    params = {"corpus_size": 10, "batch_size": 1}
    model_fn(fake_input, fake_label, params)
    with tf.Session() as sess:
        print(sess.run(fake_input_onehot))

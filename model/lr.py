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

    corpus_size = params["corpus_size"]
    learning_rate = params["learning_rate"]

    features_onehot = tf.reduce_sum(tf.one_hot(features, depth=corpus_size), axis=1)

    with tf.name_scope("lr"):
        weight_init = tf.truncated_normal(shape=[corpus_size, 1], mean=0.0, stddev=1.0)
        weight = tf.Variable(weight_init)
        bais = tf.Variable([0.0])
        y_pred = tf.sigmoid(tf.matmul(features_onehot, weight) + bais)

    tf.summary.image("weight", tf.expand_dims(tf.expand_dims(weight, axis=2), axis=0), 1)

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.estimator.inputs import numpy_input_fn
from sklearn.metrics import accuracy_score


warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', category=ImportWarning)
plt.style.use("ggplot")


def _softplus_inverse(x):
    return tf.log(tf.math.expm1(x))


tfd = tfp.distributions

WAV_SECONDS = 5
WAV_SAMPLE_RATE = 22050
WAV_SHAPE = [WAV_SECONDS * WAV_SAMPLE_RATE, 1]  # Time-steps X Features

LEARNING_RATE = 0.0005
BATCH_SIZE = 1
BASE_DEPTH = 8
LATENT_DIMENSIONS = 8
ACTIVATION = "leaky_relu"

MAX_STEPS = 201
VIZ_STEPS = 100


def make_cnn_encoder(activation, latent_size, base_depth):
    conv = functools.partial(
        tf.keras.layers.Conv1D, padding="CAUSAL", activation=activation)
    encoder_net = tf.keras.Sequential([
        conv(base_depth, 5, 1),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 1),
        conv(2 * base_depth, 5, 2),
        conv(4 * latent_size, 3, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2*latent_size, activation=None),
    ])

    def encoder(images):
        images = tf.reshape(images, (-1, WAV_SHAPE[0], 1))
        net = encoder_net(images)
        return tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(
                net[..., latent_size:] +
                _softplus_inverse(1.0)), name="code")
    return encoder


def make_cnn_decoder(activation, latent_size, output_shape, base_depth):
    conv = functools.partial(
      tf.keras.layers.Conv1D, padding="CAUSAL", activation=activation)
    decoder_net = tf.keras.Sequential([
      conv(2 * base_depth, 7, padding="VALID"),
      tf.keras.layers.UpSampling1D(size=2),
      conv(2 * base_depth, 5),
      tf.keras.layers.UpSampling1D(size=2),
      conv(2 * base_depth, 5, 2),
      tf.keras.layers.UpSampling1D(size=2),
      conv(base_depth, 5),
      tf.keras.layers.UpSampling1D(size=2),
      conv(base_depth, 5, 2),
      tf.keras.layers.UpSampling1D(size=2),
      conv(base_depth, 5),
      tf.keras.layers.UpSampling1D(size=2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(2*WAV_SHAPE[0], activation=None), ])

    def decoder(codes):
        codes = tf.reshape(codes, (-1, latent_size, 1))
        net = decoder_net(codes)
        return tfd.Normal(
            loc=net[..., :WAV_SHAPE[0]],
            scale=tf.nn.softplus(
                net[..., WAV_SHAPE[0]:] + _softplus_inverse(1.0)),
            name="wav")
    return decoder


def cnn_model_fn(features, labels, mode, params, config):
    encoder = make_cnn_encoder(
        params["activation"], params["latent_size"],
        params["base_depth"])
    decoder = make_cnn_decoder(
        params["activation"], params["latent_size"],
        WAV_SHAPE, params["base_depth"])
    latent_prior = tfd.MultivariateNormalDiag(
        loc=tf.zeros([params["latent_size"]]),
        scale_identity_multiplier=1.0)

    approx_posterior = encoder(features)
    approx_posterior_sample = approx_posterior.sample(1)
    decoder_likelihood = decoder(approx_posterior_sample)
    distortion = -decoder_likelihood.log_prob(features)
    avg_distortion = tf.reduce_mean(distortion)
    tf.summary.scalar("distortion", avg_distortion)

    approx_posterior_log_prob = approx_posterior.log_prob(
        approx_posterior_sample)
    latent_prior_log_prob = latent_prior.log_prob(approx_posterior_sample)
    rate = (approx_posterior_log_prob - latent_prior_log_prob)
    avg_rate = tf.reduce_mean(rate)
    tf.summary.scalar("rate", avg_rate)

    elbo_local = -(rate + distortion)
    elbo = tf.reduce_mean(elbo_local)
    loss = -elbo
    tf.summary.scalar("elbo", elbo)
    importance_weighted_elbo = tf.reduce_mean(
        tf.reduce_logsumexp(elbo_local, axis=0) - tf.log(tf.to_float(1)))
    tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)
    random_wav = decoder(approx_posterior.sample(16))
    tf.summary.audio("random/sample", random_wav.sample(), sample_rate=22050)
    tf.summary.audio("random/mean", random_wav.mean(), sample_rate=22050)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                          params["max_steps"])
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'encoded_sample': approx_posterior.sample(1),
                       'encoded_mean': approx_posterior.mean(),
                       'reconstructed_sample': decoder_likelihood.sample(1),
                       'reconstructed_mean': decoder_likelihood.mean(),
                       'log_likelihood': -avg_distortion, }
    else:
        predictions = None
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op,
        eval_metric_ops={"elbo": tf.metrics.mean(elbo),
                         "elbo/importance_weighted":
                         tf.metrics.mean(importance_weighted_elbo),
                         "rate": tf.metrics.mean(avg_rate),
                         "distortion": tf.metrics.mean(avg_distortion), },
        predictions=predictions, )


def train_cnn(train_input_fn):
    params = {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'latent_size': LATENT_DIMENSIONS,
        'activation': ACTIVATION,
        'base_depth': BASE_DEPTH,
        'max_steps': MAX_STEPS,
    }
    params["activation"] = getattr(tf.nn, params["activation"])
    estimator = tf.estimator.Estimator(
      cnn_model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          # model_dir=MODEL_DIR,
          save_checkpoints_steps=VIZ_STEPS,
      ),
    )
    for _ in range(MAX_STEPS // VIZ_STEPS):
        estimator.train(input_fn=train_input_fn, steps=VIZ_STEPS)
    return estimator


def predict_estimators(new_point, estimator_tuples):
    max_likelihood = None
    pred_label = None
    for label, estimator in estimator_tuples:
        pred = estimator.predict(new_point)
        likelihood = pred['log_likelihood']
        if max_likelihood is None or likelihood > max_likelihood:
            pred_label = label
            max_likelihood = likelihood
    return pred_label


def fit_cnn_vae(train_data):
    estimators = []
    for label in train_data['label'].unique():
        print("Training VAE for label: {}".format(label))
        x_train = train_data.loc[train_data['label']
                                 == label].copy()
        x_train.pop('label')
        train_input_fn = numpy_input_fn(
            x_train.values.astype(np.float32),
            shuffle=True,
            batch_size=1
        )
        estimator = train_cnn(train_input_fn)
        estimators.append((label, estimator))
    return estimators


def single_vae(train_data, test_data):
    x_train = train_data.drop(['label'], axis=1)
    x_test = test_data.drop(['label'], axis=1)
    train_input_fn = numpy_input_fn(
        x_train.values.astype(np.float32),
        shuffle=True,
        batch_size=1)
    estimator = train_cnn(train_input_fn)
    predict_input_fn = numpy_input_fn(
        x_test.values.astype(np.float32),
        shuffle=False,
        batch_size=1)
    predictions = list(
        estimator.predict(input_fn=predict_input_fn,
                          yield_single_examples=False))
    predicted_encodings = pd.DataFrame([predictions[i]
                                        ['encoded_sample'][0][0]
                                        for i in range(
                                            len(predictions))])
    return(predictions, predicted_encodings)


def create_y_testdf(test_data):
    y_test = test_data['label']
    y_df = y_test.to_frame()
    y_df['label_cat'] = y_df['label'].astype('category')
    y_df['label_int'] = y_df['label_cat'].cat.codes
    return y_df


def predict_from_estimators(test_point, estimator_tuples):
    max_likelihood = None
    pred_label = None
    ex = np.array([test_point.astype(np.float32)])
    predict_input_fn = numpy_input_fn(
        ex,
        shuffle=False,
        batch_size=1
    )
    for label, estimator in estimator_tuples:
        pred = list(estimator.predict(predict_input_fn,
                                      yield_single_examples=False))[0]
        likelihood = pred['log_likelihood']
        if max_likelihood is None or likelihood > max_likelihood:
            pred_label = label
            max_likelihood = likelihood
    return pred_label


def mixture_vae(test_data, estimators):
    x_test = test_data.drop(['label'], axis=1)
    predictions = []
    for data_point in np.array(x_test):
        pr = predict_from_estimators(data_point, estimators)
        predictions.append(pr)
    return predictions


def mixvae_accuracy(test_data, predictions):
    y_test = test_data['label']
    accuracy = accuracy_score(y_test, predictions)
    print('CNN VAE Mixture Classification Accuracy:{}%'.format(
        accuracy*100))

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed


plt.style.use("ggplot")


TrainingResults = namedtuple('TraingResults',
                             ['qw', 'qz', 'w_mean', 'w_stddev',
                              'z_mean', 'z_stddev', 'generated_x_samples'])


def train_ppca(x_train, data_dim, latent_dim, num_datapoints,
               stddv_datapoints):
    def probabilistic_pca(data_dim, latent_dim, num_datapoints,
                          stddv_datapoints):
        w = ed.Normal(loc=tf.zeros([data_dim, latent_dim]),
                      scale=2.0 * tf.ones([data_dim, latent_dim]),
                      name="w")  # PRINCIPAL COMPONENTS
        z = ed.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                      scale=tf.ones([latent_dim, num_datapoints]),
                      name="z")  # LATENT VARIABLE
        x = ed.Normal(loc=tf.matmul(w, z),
                      scale=stddv_datapoints * tf.ones(
                      [data_dim, num_datapoints]), name="x")
        return x, (w, z)

    log_joint = ed.make_log_joint_fn(probabilistic_pca)

    def target(w, z):
        """Unnormalized target density"""
        return log_joint(data_dim=data_dim,
                         latent_dim=latent_dim,
                         num_datapoints=num_datapoints,
                         stddv_datapoints=stddv_datapoints,
                         w=w, z=z, x=x_train)

    tf.reset_default_graph()

    def variational_model(qw_mean, qw_stddv, qz_mean, qz_stddv):
        qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
        qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
        return qw, qz

    log_q = ed.make_log_joint_fn(variational_model)

    def target_q(qw, qz):
        return log_q(qw_mean=qw_mean, qw_stddv=qw_stddv,
                     qz_mean=qz_mean, qz_stddv=qz_stddv,
                     qw=qw, qz=qz)
    qw_mean = tf.Variable(np.ones([data_dim, latent_dim]),
                          dtype=tf.float32)
    qz_mean = tf.Variable(np.ones([latent_dim, num_datapoints]),
                          dtype=tf.float32)
    qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones(
        [data_dim, latent_dim]), dtype=tf.float32))
    qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones(
        [latent_dim, num_datapoints]), dtype=tf.float32))
    qw, qz = variational_model(qw_mean=qw_mean, qw_stddv=qw_stddv,
                               qz_mean=qz_mean, qz_stddv=qz_stddv)
    energy = target(qw, qz)
    entropy = -target_q(qw, qz)
    elbo = energy + entropy
    optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
    train = optimizer.minimize(-elbo)
    init = tf.global_variables_initializer()
    t = []
    num_epochs = 100
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_epochs):
            sess.run(train)
            if i % 5 == 0:
                t.append(sess.run([elbo]))
        w_mean_inferred = sess.run(qw_mean)
        w_stddv_inferred = sess.run(qw_stddv)
        z_mean_inferred = sess.run(qz_mean)
        z_stddv_inferred = sess.run(qz_stddv)

    plt.plot(range(1, num_epochs, 5), t)
    plt.show()

    def replace_latents(w=None, z=None):
        """
        Helper function that replaces our prior for w and z
        """

        def interceptor(rv_constructor, *rv_args, **rv_kwargs):
            """Replaces the priors with actual values"""
            name = rv_kwargs.pop("name")
            if name == "w":
                rv_kwargs["value"] = w
            elif name == "z":
                rv_kwargs["value"] = z
            return rv_constructor(*rv_args, **rv_kwargs)

        return interceptor

    with ed.interception(replace_latents(w_mean_inferred,
                                         z_mean_inferred)):
        generate = probabilistic_pca(data_dim=data_dim,
                                     latent_dim=latent_dim,
                                     num_datapoints=num_datapoints,
                                     stddv_datapoints=stddv_datapoints)

    with tf.Session() as sess:
        x_generated, _ = sess.run(generate)

    return TrainingResults(qw, qz, w_mean_inferred,
                           w_stddv_inferred,
                           z_mean_inferred,
                           z_stddv_inferred,
                           x_generated)


latent_dim = 50
stddv_datapoints = 1
training_results = []

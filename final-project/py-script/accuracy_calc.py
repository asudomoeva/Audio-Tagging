import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.distributions as tfd


def calculate_probability(training_results, number_labels,
                          latent_dim, stddv_datapoints, X_test):
    final_proba = []
    for i in range(number_labels):
        z = list(pd.DataFrame(training_results[i][1]
                              .z_mean.transpose()).mean())
        z_av = np.array(z, dtype="float32")
        z_av.shape = (1, latent_dim)
        x_dist = tfd.Normal(loc=tf.matmul(z_av,
                                          training_results[i][1]
                                          .w_mean.transpose()),
                            scale=stddv_datapoints *
                            tf.ones([1,
                                     training_results[i][1]
                                     .w_mean.shape[0]]),
                            name="x_experiment{}".format(i))
        proba = []
        for testpoint in X_test:
            probability = tf.reduce_mean(x_dist.log_prob(testpoint)).eval()
            proba.append(probability)
        final_proba.append(proba)
    return np.array(final_proba).transpose()


def assign_class(final_probability_list):
    assigned_labels = []
    for point in final_probability_list:
        value = np.argmax(point)
        assigned_labels.append(value)
    return assigned_labels

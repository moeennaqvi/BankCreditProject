import numpy as np
import tensorflow as tf


def sgd_step(x, y, w, delta_P, optimizer, lmbda=0.01):
    """Perform one step of SGD.

    Arguments:
    ----------
        x: Feature vector.
        y: Ground truth.
        w: Feature parameters to be learned.
        delta_p: Difference in probability of repaying loan when
            including/excluding sensitive variables.
        optimizer: A TensorFlow optimizer.
        lmbda: Regularization parameter.

    Returns:
    --------
        Updated estimate of parameter vector w.
    """

    # NOTE: TF is sensitive about dtypes.
    x = tf.cast(x, dtype=DTYPE)
    w = tf.cast(w, dtype=DTYPE)
    lmbda = tf.cast(lmbda, dtype=DTYPE)
    delta_P = tf.cast(delta_P, dtype=DTYPE)

    with tf.GradientTape() as tape:

        # Just in case.
        tape.watch(w)

        # Parametrized policy.
        pi_w = tf.exp(tf.matmul(w, x)) / (tf.exp(tf.matmul(w, x)) + 1)

        # NOTE: Maximize V <=> minimize -1 * V.
        V = (lmbda - 1) * 1 / tf.exp(tf.square(y - pi_w)) + lmbda * tf.reduce_sum(pi_w * delta_P)

    optimizer.minimize(V, [w])

    # Cast to numpy so can be mixed with Python objects.
    return w.numpy()


def experiment(X, y, num_epochs=1000):
    """

    Arguments:
    ----------
        X: Feature matrix of N samples x P features.
        y: Ground truths.
        num_epochs: Number of SGD steps.

    Returns:
    --------
           The selected policy parameterization.
    """

    np.random.seed(seed)

    # Initial parameter estimate and optimization object.
    w_i = np.random.random(X.shape[1])
    w_i_tf = tf.Variable(w_i, dtype=DTYPE)

    # Train each of the RF models.
    rf_x.train(X, y)
    rf_xz.train(X, y)

    # Deviation from the RF model being independent on sensitive variable z.
    delta_p = (rf_x.predict_proba(X) - rf_xz.predict_proba(X)) ** 2

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    for _ in range(num_epochs):

        i = np.random.choice(np.arange(X.shape[0]))
        w_i = sgd_step(x=X[i], y=y[i], w=w_i_tf, delta_P=delta_p, optimizer=optimizer)

    return w_i


if __name__ == "__main__":
    experiment()

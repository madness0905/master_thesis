import tensorflow as tf
import datetime
from tensorflow.keras.layers import Input


class LinearClassifier(tf.keras.Model):

    def __init__(
        self,
        batch_size,
        input_shape=None,
        num_classes=2,
    ):
        super().__init__()
        if num_classes == 2:
            self.dense = tf.keras.layers.Dense(
                units=1,
                activation='sigmoid',
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer(),
                input_shape=(batch_size, input_shape))
        else:
            self.dense = tf.keras.layers.Dense(
                units=num_classes,
                activation='softmax',
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer(),
                input_shape=(None, input_shape))

    def call(self, inputs):
        # input = tf.expand_dims(input, axis=-1)

        output = self.dense(inputs)
        return output

    def save_model_weights(self, name):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_weights(name + current_time)


if __name__ == '__main__':
    model = LinearClassifier(num_classes=2)
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print(model.variables)

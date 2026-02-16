import cv2
import numpy as np

from other.root import *
from tensorflow.python.keras import layers, models


def decoder_block(inputs, num_filters, name):
    x = layers.Conv2DTranspose(num_filters, (3, 3), padding='same', name=f'{name}_conv1')(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)
    x = layers.Conv2DTranspose(num_filters, (3, 3), padding='same', name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.ReLU(name=f'{name}_relu2')(x)
    # x = layers.MaxPooling2D((2, 2))(x)
    return x


def LinkNetClassification_(num_classes, input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    # encoder #
    encoder1 = encoder_block(inputs, 32, 'enc1')
    encoder2 = encoder_block(encoder1, 64, 'enc2')
    encoder3 = encoder_block(encoder2, 128, 'enc3')
    encoder4 = encoder_block(encoder3, 256, 'enc4')
    # decoder #
    decoder1 = decoder_block(encoder4, 256, 'dec4')
    decoder2 = decoder_block(decoder1, 128, 'dec3')
    decoder3 = decoder_block(decoder2, 64, 'dec2')
    decoder4 = decoder_block(decoder3, 32, 'dec1')

    x = layers.GlobalAveragePooling2D()(decoder4)

    # Classification output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name='linknet_classification')


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import Ones, Zeros


class CustomBatchNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, momentum=0.99, **kwargs):
        super(CustomBatchNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(shape=(dim,),
                                     initializer=Ones(),
                                     trainable=True,
                                     name='gamma')
        self.beta = self.add_weight(shape=(dim,),
                                    initializer=Zeros(),
                                    trainable=True,
                                    name='beta')
        self.moving_mean = self.add_weight(shape=(dim,),
                                           initializer=Zeros(),
                                           trainable=False,
                                           name='moving_mean')
        self.moving_variance = self.add_weight(shape=(dim,),
                                               initializer=Ones(),
                                               trainable=False,
                                               name='moving_variance')

    def call(self, inputs, training=None):
        if training:
            mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
            self.moving_mean.assign(self.moving_mean * self.momentum + mean * (1 - self.momentum))
            self.moving_variance.assign(self.moving_variance * self.momentum + variance * (1 - self.momentum))
        else:
            mean = self.moving_mean
            variance = self.moving_variance
        c = 1.8498;
        k = 3;
        x = 1
        ck = np.power(c, k)
        self.epsilon = k / ck * np.power(x, 2) * np.exp(np.power((-x / c), k))
        return tf.nn.batch_normalization(inputs, mean, variance, self.beta, self.gamma, self.epsilon)


# Example encoder block using CustomBatchNormalization
def encoder_block(inputs, num_filters, name):
    # First convolutional layer
    x = layers.Conv2D(num_filters, (3, 3), padding='same', name=f'{name}_conv1')(inputs)
    x = CustomBatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)

    # Second convolutional layer
    x = layers.Conv2D(num_filters, (3, 3), padding='same', name=f'{name}_conv2')(x)
    x = CustomBatchNormalization(name=f'{name}_bn2')(x)
    x = layers.ReLU(name=f'{name}_relu2')(x)

    # Optionally, you can add a MaxPooling layer here for downsampling
    # x = layers.MaxPooling2D((2, 2))(x)

    return x


def reshape_(data):
    # Convert the data to the appropriate data type (e.g., uint8)
    data_uint8 = (data * 255).astype(np.uint8)
    # Reshape and resize the data
    resized_data = np.zeros((len(data), 32, 32, 3), dtype=np.uint8)
    for i in range(len(data_uint8)):
        print('Reshape:', i)
        img = data_uint8[i].reshape(1, data_uint8.shape[1])  # Reshape each feature for processing
        img_resized = cv2.resize(img, (32, 32))  # Resize the feature to the desired shape
        img_resized_3d = np.repeat(img_resized[:, :, np.newaxis], 3, axis=2)  # Convert to 3 channels
        resized_data[i] = img_resized_3d  # Assign the resized image to the corresponding index in the new array
    return resized_data


def hybrid_loss(ytrue, ypred):
    import numpy as np

    def triplet_loss(anchor, positive, negative, margin=1.0):
        """
        Compute triplet loss manually.

        Parameters:
        - anchor (numpy array): Anchor embeddings.
        - positive (numpy array): Positive embeddings.
        - negative (numpy array): Negative embeddings.
        - margin (float): The margin for the loss.

        Returns:
        - loss (float): Computed triplet loss.
        """
        # Compute distances
        pos_distance = np.sum((anchor - positive) ** 2, axis=1)
        neg_distance = np.sum((anchor - negative) ** 2, axis=1)

        # Compute loss
        loss = np.maximum(pos_distance - neg_distance + margin, 0.0)
        return np.mean(loss)

    # loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    anchor = ytrue
    positive = ytrue  # Here using ytrue as positive, adapt as needed
    negative = ypred  # Here using ypred as negative, adapt as needed

    # Compute triplet loss
    loss2 = triplet_loss(anchor, positive, negative)
    # # Compute beta
    # indices = tf.where(ytrue != 0)
    # ind = tf.shape(indices)[0]
    # try:beta = tf.cast(tf.shape(ytrue)[0], tf.float32) / (tf.cast(ind, tf.float32) * tf.cast(tf.shape(ytrue)[0], tf.float32))
    # except:beta=1.0
    # loss1 = loss_object(ytrue, ypred)
    # # Combine losses
    # hybrid_loss = loss1 + beta * loss2
    return loss2


def prop_linknet(x_train, x_test, y_train, y_test):
    print("linknet")
    ln = len(np.unique(y_train))
    x_train = np.resize(x_train, (x_train.shape[0], 32, 32, 3))
    x_test = np.resize(x_test, (x_test.shape[0], 32, 32, 3))
    # Instantiate the LinkNet-like classification model
    num_classes = ln  # Define the number of classes for your classification task
    input_shape = (32, 32, 3)
    model = LinkNetClassification_(num_classes, input_shape)
    # Compile the model with appropriate loss and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=hybrid_loss, metrics=['accuracy'])
    model.summary()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    # Train the model
    model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=0)
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict, axis=1)
    # from tensorflow.keras.models import save_model
    # save_model(model, 'pyra.h5')
    y_pred = array(y_predict, axis=[0, 1])
    return y_pred


def linknet(x_train, x_test, y_train, y_test):
    print("linknet")
    ln = len(np.unique(y_train))
    x_train = np.resize(x_train, (x_train.shape[0], 32, 32, 3))
    x_test = np.resize(x_test, (x_test.shape[0], 32, 32, 3))
    num_classes = ln  # Define the number of classes for your classification task
    input_shape = (32, 32, 3)
    model = LinkNetClassification_(num_classes, input_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=0)
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict, axis=1)
    y_pred = array(y_predict, axis=[0, 1])
    return y_pred

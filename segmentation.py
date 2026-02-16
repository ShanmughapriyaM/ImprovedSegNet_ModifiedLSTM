import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model


def segnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)

    # Decoder
    x = UpSampling2D()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def custom_loss(y_true, y_pred):
    # Compute the height and width of the image
    H = tf.cast(tf.shape(y_true)[1], tf.float32)
    W = tf.cast(tf.shape(y_true)[2], tf.float32)

    # Flatten the labels and predictions
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)

    # Calculate the custom loss
    loss = -2 * y_true_flat * tf.math.log(y_pred_flat + 1e-9) \
           - (1 - y_true_flat) * tf.math.log(1 - y_pred_flat + 1e-9) \
           + (tf.math.square(y_true_flat) / (H * W)) * tf.math.log(y_pred_flat + 1e-9) \
           - (y_true_flat * (1 - y_true_flat) / (H * W)) * tf.math.log(1 - y_pred_flat + 1e-9)

    return tf.reduce_mean(loss)


# Adjust input shape according to your data
input_shape = (256, 256, 3)

# Adjust based on the number of classes in your segmentation task
num_classes = 3

# Create the SegNet model
seg_model = segnet(input_shape, num_classes)

# Compile the model with the custom loss function
seg_model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

# Load or create an example image for prediction
# Ensure 'im' is an array of shape (256, 256, 3) with proper normalization
# Example placeholder for im:
im = np.random.rand(256, 256, 3)  # Replace this with your actual image

# Make prediction
seg = seg_model.predict(im[np.newaxis, :, :, :])

# Display prediction
print("Segmentation result shape:", seg.shape)
print("Segmentation result:", seg)



def segnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    # Encoder
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)

    # Decoder
    x = UpSampling2D()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
input_shape = (256, 256, 3)  # Adjust input shape according to your data
num_classes = ln  # Adjust based on the number of classes in your segmentation task
seg_model = segnet(input_shape, 3)
seg = seg_model.predict(im[np.newaxis, :, :, :])
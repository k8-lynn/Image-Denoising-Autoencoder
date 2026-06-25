import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 3. Add random noise to the images
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# Clip the values to ensure they stay strictly between 0 and 1
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

# Build the Autoencoder model
autoencoder = Sequential([
    # ENCODER: Flatten the 28x28 image and compress it down to a tiny 64-node representation
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    
    # DECODER: Expand it back up to the original 784 pixels (28x28)
    Dense(128, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

print("Starting training...")
# Train the model
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# Test it on a few unseen images
decoded_imgs = autoencoder.predict(x_test_noisy)

# Visualize the Before and After
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # Display noisy images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].numpy(), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Display reconstructed (clean) images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i], cmap='gray')
    plt.title("Cleaned")
    plt.axis('off')

plt.tight_layout()
plt.show()
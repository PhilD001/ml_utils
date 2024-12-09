import tensorflow as tf
from processing.processing import load_keras_sample_time_series_db

# Define a function to add 1 to each value. This is where you would cut off NaNs?
def add_one(x):
    return x + 1


# Create a Lambda layer using the defined function
add_one_layer = tf.keras.layers.Lambda(lambda x: add_one(x))

# Example time series data as a tensor
time_series_data, y, user, channel_names, meta_data = load_keras_sample_time_series_db()


# Apply the preprocessing layer to the data
processed_data = add_one_layer(time_series_data)

# Add a simple CNN classifier
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(time_series_data.shape[0], 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification example
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate some random labels for demonstration
labels = tf.random.uniform((time_series_data.shape[0], 1), maxval=2, dtype=tf.int32)

# Train the model
model.fit(processed_data, labels, epochs=5)
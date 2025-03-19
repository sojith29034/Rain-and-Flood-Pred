import tensorflow as tf
import tensorflowjs as tfjs

# Load the model
model = tf.keras.models.load_model("predictions.keras")

# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(model, "tfjs_model")

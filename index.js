const tf = require('@tensorflow/tfjs');

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
require('@tensorflow/tfjs-node');

// Train a simple model:
const model = tf.sequential();
// Dense layer => fully connected layer => every node in this layer is connected to every node in the previous layer 
const hidden = tf.layers.dense();

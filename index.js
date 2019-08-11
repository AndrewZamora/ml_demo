const tf = require('@tensorflow/tfjs-node');

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU. (This only works with linux)

// // Train a simple model:
const model = tf.sequential();
// // Dense layer => fully connected layer => every node in this layer is connected to every node in the previous layer 
const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
});
const output = tf.layers.dense({
    units: 3,
    inputShape: [4],
    activation: 'sigmoid'
});

model.add(hidden);
model.add(output);

// optimizer will minimize loss function
const sgdOptimizer = tf.train.sgd(0.1);

model.compile({
    optimizer: sgdOptimizer,
    loss: 'meanSquaredError'
});
const tf = require('@tensorflow/tfjs-node');
// Use '@tensorflow/tfjs-node-gpu' if running with GPU. (This only works with linux)

// This is a simple model
const model = tf.sequential();
// Create the hidden layer (Dense layer => fully connected layer => every node in this layer is connected to every node in the previous layer)
const hidden = tf.layers.dense({
    units: 4,   // number of nodes
    inputShape: [2], // input shape
    activation: 'sigmoid'
});
// Add the hidden layer
model.add(hidden);
// Create another layer
const output = tf.layers.dense({
    units: 1,
    // input shape is inferred from the previous layer
    activation: 'sigmoid'
});
model.add(output);
// optimizer will minimize loss function (This case gradient descent)
const sgdOptimizer = tf.train.sgd(0.5);

model.compile({
    optimizer: sgdOptimizer,
    loss: tf.losses.meanSquaredError
});

const xs = tf.tensor2d([
    [0, 0],
    [0.5, 0.5],
    [1, 1],
]);

const ys = tf.tensor2d([
    [1],
    [0.05],
    [0],
]);

async function train() {
    for (let i = 0; i < 1000; i++) {
        const config = {
            shuffle: true,
            epochs: 10
        }
        const response = await model.fit(xs, ys, config);
        console.log(response.history.loss[0])
    }

}

train().then(() => {
    let outputs = model.predict(xs);
    outputs.print();
});




// const xs = tf.tensor2d([
//     [0.25, 0.92],
//     [0.12, 0.3],
//     [0.4, 0.74],
//     [0.1, 0.22],
// ]);

// let ys = model.predict(inputs);

// outputs.print();

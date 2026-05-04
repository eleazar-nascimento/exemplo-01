import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
}

const normalize = (value, min, max) => (value - min) / (max - min) || 1;

function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    const ageMin = Math.min(...ages);
    const ageMax = Math.max(...ages);

    const priceMin = Math.min(...prices);
    const priceMax = Math.max(...prices);

    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];
    
    const colorsIndex = Object.fromEntries(colors.map((color, index) => [color, index]))
    const categoryIndex = Object.fromEntries(categories.map((category, index) => [category, index]))    

     // computar a média de idade dos comprados por produto
     // ajua a personalizar

    const midAge = (ageMin + ageMax) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        user.purchases.forEach(product => {
            ageSums[product.name] = (ageSums[product.name] || 0) + user.age;
            ageCounts[product.name] = (ageCounts[product.name] || 0) + 1;
        })
    });

    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageSums[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge;
            
            
            return [product.name, normalize(avg, ageMin, ageMax)];
        })
    );

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex: categoryIndex,
        productAvgAgeNorm,
        ageMin,
        ageMax,
        priceMin,
        priceMax,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + colors + categories
        dimentions: 2 + categories.length + colors.length, // age, price + one-hot categories + one-hot colors
    }
}

const oneHotWeighted = (index, lenght, weight) =>
    tf.oneHot(index, lenght).cast('float32').mul(weight)

function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalize(
            product.price, 
            context.priceMin, 
            context.priceMax
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

   return tf.concat1d([price, age, category, color]);


}

function encodeUser(user, context) {
    // criar um vetor de características para o usuário, baseado nos produtos comprados
    // por exemplo, a média dos vetores dos produtos comprados

    if(user.purchases.length) {
        return tf.stack(
            user.purchases.map(product => encodeProduct(product, context))
        ).mean(0)
        .reshape([1, context.dimentions]);
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado
            tf.tensor1d([normalize(user.age, context.ageMin, context.ageMax) * WEIGHTS.age]), // idade normalizada
            tf.zeros([context.numCategories]), // categorias são ignoradas
            tf.zeros([context.numColors]), // cores são ignoradas
        ]
    ).reshape([1, context.dimentions]);
}

function createTrainingData(context) {
    const inputs = [];
    const labels = [];

    context.users
    .filter(user => user.purchases.length)
    .forEach(user => {
        const userVector = encodeUser(user, context).dataSync();

        context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync();

            const label = user.purchases.some(p => p.name === product.name) ? 1 : 0;

            //combinar usuario + product
            inputs.push([...userVector, ...productVector]);
            labels.push(label);
        })
    })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2,
        // tamanho = userVector + productVector
    }

    // para cada usuário, pegar os produtos comprados e criar um vetor de características
    // usar esses vetores para treinar um modelo de recomendação
}

async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential();

    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: comeca a comprimir a informação)
    // - activation relu (introduz não linearidade, ajudando a modelar relações complexas)

    model.add(tf.layers.dense({ inputShape: [trainData.inputDimention], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 10,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    const products = await (await fetch('/data/products.json')).json();
    
    const context = makeContext(products, users)

    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync(),
        }
    })

    _globalCtx = context;

    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData);
 
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}
function recommend(user, ctx) {
    if(!_model) {
        console.warn('Model not trained yet');
        return;
    }

    const context = _globalCtx

    const userVector = encodeUser(user, _globalCtx).dataSync();

    // Em Aplicações reais:
    // Armazene todos os vetores de produtos em um banco de dados
    // vetorial (como Postgres, Neo4j ou Pinecone), CromaDB
    // Consulta: Encontre os 200 produtos mais próximos dos usuário
    // Execute _model.predict() apenas nesses produtos

    // crie pares de entrada: para cada produto, concatene o 
    // vetor do usuario
    //.  com o vetor codificado do produto.
    //   por quê? O modelo prevê o "score de compatibilidade"
    // para cada par (usuario, produto).

    const inputs = context.productVectors.map(({vector}) => {
        return [...userVector, ...vector];
    })

    // Converta todos esses pares (usuarios, produto) em um unico tensor.
    // Formato: [numProdutos, inputDimention]

    const inputTensor = tf.tensor2d(inputs);

    // Rode a rede neural treinada em todos os pares (usuario, produto) para obter uma vez.
        // O resultado é uma pontuação para cada produto em entre 0 e 1.
    
        // Quanto maior, maior a probabilidade do usuário querer aquele produto.

    const predictions = _model.predict(inputTensor).dataSync();
        // Extraia as pontuações para un array Js normal.

    const scores = predictions
    const recommendations = context.productVectors.map((product, index) => ({
        ...product.meta,
        name: product.name,
        score: scores[index], // previsão do modelo para estre produto
    }))

    const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score);

    
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedRecommendations
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};

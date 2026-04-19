import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
}

const normalize = (value, min, max) => (value - min) / (max - min) || 1;

function makeContext(catalog, users) {
    const ages = users.map(u => u.age);
    const prices = catalog.map(p => p.price);

    const ageMin = Math.min(...ages);
    const ageMax = Math.max(...ages);

    const priceMin = Math.min(...prices);
    const priceMax = Math.max(...prices);

    const colors = [...new Set(catalog.map(p => p.color))];
    const categories = [...new Set(catalog.map(p => p.category))];
    
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
        catalog.map(product => {
            const avg = ageSums[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge;
            
            
            return [product.name, normalize(avg, ageMin, ageMax)];
        })
    );

    return {
        catalog,
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

async function trainModel({ users }) {
    console.log('Training model with users:', users)
    const catalog = await (await fetch('/data/products.json')).json();
    
    const context = makeContext(catalog, users)

    context.productVectors = catalog.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync(),
        }
    })

    debugger

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);

}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};

# E-commerce Recommendation System

A web application that displays user profiles and product listings, with a TensorFlow.js-based recommendation engine and **ChromaDB** vector database for pre-filtering products by similarity.

## Project Structure

- `index.html` - Main HTML file for the application
- `src/index.js` - Entry point for the application
- `src/view/` - Contains classes for managing the DOM and templates
- `src/controller/` - Contains controllers to connect views and services
- `src/service/` - Contains business logic and ChromaDB integration
- `src/workers/` - Web Worker for TensorFlow.js model training and inference
- `data/` - Contains JSON files with user and product data (100 products)
- `docker-compose.yml` - Docker Compose for ChromaDB

## Setup and Run

### 1. Start ChromaDB (requires Docker)

```bash
docker compose up -d
```

This starts a ChromaDB instance on `http://localhost:8000`.

### 2. Install dependencies

```bash
npm install
```

### 3. Start the application

```bash
npm start
```

### 4. Open your browser and navigate to `http://localhost:3000`

## Architecture

```
Browser (Web Worker + TensorFlow.js)
  │
  ├── Treina modelo com dados de compras dos usuários
  ├── Codifica produtos e usuários em vetores
  ├── Armazena vetores no ChromaDB (via REST API)
  │
  └── Na recomendação:
      1. ChromaDB pré-filtra os top 20 produtos mais similares (cosine similarity)
      2. model.predict() roda apenas nos 20 candidatos
      3. Produtos ordenados por score de compatibilidade
```

## Features

- User profile selection with details display
- Past purchase history display
- Product listing with "Buy Now" functionality
- Purchase tracking using sessionStorage
- **TensorFlow.js neural network** for recommendation scoring
- **ChromaDB vector database** for similarity pre-filtering
- Automatic fallback when ChromaDB is unavailable
- ChromaDB connection status indicator in the UI

## ChromaDB Integration

The system uses ChromaDB as a vector database to optimize the recommendation pipeline:

| Without ChromaDB | With ChromaDB |
|---|---|
| `predict()` runs on all 100 products | ChromaDB pre-filters top 20 by cosine similarity |
| O(N) predictions | O(K) predictions where K << N |
| Works for small catalogs | Scales to thousands of products |

If ChromaDB is unavailable, the system automatically falls back to analyzing all products.

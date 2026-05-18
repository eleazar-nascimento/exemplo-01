/**
 * ChromaService - Comunicação com ChromaDB via REST API (v1)
 * 
 * Funciona diretamente no Web Worker via fetch, sem dependências externas.
 * Armazena vetores de produtos e realiza buscas por similaridade vetorial.
 */

const CHROMA_BASE_URL = 'http://localhost:8000';
const CHROMA_API = `${CHROMA_BASE_URL}/api/v1`;
const COLLECTION_NAME = 'product_vectors';

let _collectionId = null;

async function chromaFetch(path, options = {}) {
    const url = `${CHROMA_API}${path}`;
    try {
        const response = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });

        if (!response.ok) {
            const error = await response.text();
            console.warn(`[ChromaService] HTTP ${response.status} em ${path}:`, error);
            return null;
        }

        // Alguns endpoints retornam 204 sem body
        const text = await response.text();
        return text ? JSON.parse(text) : true;
    } catch (error) {
        console.warn('[ChromaService] Erro de rede:', error.message);
        return null;
    }
}

/**
 * Verifica se o ChromaDB está acessível
 */
export async function isAvailable() {
    try {
        const result = await chromaFetch('/heartbeat');
        return result !== null;
    } catch {
        return false;
    }
}

/**
 * Cria ou obtém a collection de vetores de produtos usando get_or_create
 */
export async function getOrCreateCollection() {
    const result = await chromaFetch('/collections', {
        method: 'POST',
        body: JSON.stringify({
            name: COLLECTION_NAME,
            metadata: { 'hnsw:space': 'cosine' },
            get_or_create: true,
        }),
    });

    if (!result || !result.id) {
        console.warn('[ChromaService] Falha ao criar/obter collection');
        return null;
    }

    _collectionId = result.id;
    console.log(`[ChromaService] Collection pronta: ${_collectionId}`);
    return _collectionId;
}

/**
 * Limpa todos os vetores da collection existente
 * Ao invés de deletar e recriar (que tem bug no 0.6.3),
 * recria com get_or_create que reutiliza a existente
 */
export async function resetCollection() {
    // Simplesmente recria — o get_or_create retorna a existente
    await getOrCreateCollection();
}

/**
 * Armazena os vetores codificados dos produtos no ChromaDB
 * Usa upsert para sobrescrever vetores existentes
 * 
 * @param {Array} productVectors - Array de { name, meta, vector }
 */
export async function addProductVectors(productVectors) {
    if (!_collectionId) {
        console.warn('[ChromaService] Collection não inicializada');
        return false;
    }

    const ids = productVectors.map(p => `product_${p.meta.id}`);
    const embeddings = productVectors.map(p => Array.from(p.vector));
    const metadatas = productVectors.map(p => ({
        name: p.name,
        category: p.meta.category,
        price: p.meta.price,
        color: p.meta.color,
        product_id: p.meta.id,
    }));

    // Usar upsert para atualizar vetores existentes sem erro de duplicata
    const result = await chromaFetch(
        `/collections/${_collectionId}/upsert`,
        {
            method: 'POST',
            body: JSON.stringify({ ids, embeddings, metadatas }),
        }
    );

    if (result !== null) {
        console.log(`[ChromaService] ${productVectors.length} vetores armazenados no ChromaDB`);
        return true;
    }

    return false;
}

/**
 * Busca os N produtos mais similares ao vetor do usuário
 * 
 * @param {Float32Array|Array} userVector - Vetor codificado do usuário
 * @param {number} nResults - Quantidade de candidatos a retornar
 * @returns {Array|null} - Array de product_ids ordenados por similaridade, ou null se indisponível
 */
export async function querySimilar(userVector, nResults = 20) {
    if (!_collectionId) {
        console.warn('[ChromaService] Collection não inicializada');
        return null;
    }

    const result = await chromaFetch(
        `/collections/${_collectionId}/query`,
        {
            method: 'POST',
            body: JSON.stringify({
                query_embeddings: [Array.from(userVector)],
                n_results: nResults,
                include: ['metadatas', 'distances'],
            }),
        }
    );

    if (!result || !result.ids || !result.ids[0]) return null;

    const candidates = result.ids[0].map((id, index) => ({
        productId: result.metadatas[0][index].product_id,
        name: result.metadatas[0][index].name,
        distance: result.distances[0][index],
    }));

    console.log(`[ChromaService] ${candidates.length} candidatos retornados pelo ChromaDB`);
    return candidates;
}

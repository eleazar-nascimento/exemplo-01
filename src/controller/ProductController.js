export class ProductController {
    #productView;
    #currentUser = null;
    #events;
    #productService;
    constructor({
        productView,
        events,
        productService
    }) {
        this.#productView = productView;
        this.#productService = productService;
        this.#events = events;
        this.init();
    }

    static init(deps) {
        return new ProductController(deps);
    }

    async init() {
        this.setupCallbacks();
        this.setupEventListeners();
        const products = await this.#productService.getProducts();
        this.#productView.render(products, true);
    }

    setupEventListeners() {

        this.#events.onUserSelected((user) => {
            this.#currentUser = user;
            this.#productView.onUserSelected(user);
            this.#events.dispatchRecommend(user)
        })

        this.#events.onRecommendationsReady(({ recommendations, chromaUsed, totalProducts, candidatesFiltered }) => {
            this.#productView.render(recommendations, false);

            // Atualizar stats do ChromaDB na UI
            const statsEl = document.querySelector('#chromaStats');
            if (statsEl) {
                if (chromaUsed) {
                    statsEl.innerHTML = `
                        <small class="text-muted">
                            <i class="bi bi-funnel"></i> 
                            ChromaDB: ${candidatesFiltered}/${totalProducts} produtos pré-filtrados
                        </small>`;
                } else {
                    statsEl.innerHTML = `
                        <small class="text-muted">
                            <i class="bi bi-collection"></i> 
                            Analisando todos os ${totalProducts} produtos
                        </small>`;
                }
            }
        });
    }

    setupCallbacks() {
        this.#productView.registerBuyProductCallback(this.handleBuyProduct.bind(this));
    }

    async handleBuyProduct(product) {
        const user = this.#currentUser;
        this.#events.dispatchPurchaseAdded({ user, product });
    }

}

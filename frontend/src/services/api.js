/**
 * Norman API Service
 * Handles communication with the FastAPI backend
 */

const API_BASE = '/api';

/**
 * Send a chat query to the RAG system
 * @param {string} query - User's question in Vietnamese
 * @param {number} topK - Number of documents to retrieve (default: 5)
 * @returns {Promise<Object>} Chat response with answer and sources
 */
export async function chat(query, topK = 5) {
    const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query,
            top_k: topK
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || 'Chat request failed');
    }

    return response.json();
}

/**
 * Search for relevant legal documents without LLM generation
 * @param {string} query - Search query
 * @param {number} topK - Number of results (default: 5)
 * @returns {Promise<Object>} Search results
 */
export async function search(query, topK = 5) {
    const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query,
            top_k: topK
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || 'Search request failed');
    }

    return response.json();
}

/**
 * Check backend health status
 * @returns {Promise<Object>} Health status
 */
export async function healthCheck() {
    const response = await fetch(`${API_BASE}/health`);

    if (!response.ok) {
        throw new Error('Backend is not available');
    }

    return response.json();
}

export default {
    chat,
    search,
    healthCheck,
};

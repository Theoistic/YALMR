/**
 * YALMR inference client
 * Wraps the /v1/sessions family of endpoints and exposes a simple
 * async/streaming API for use in browser-based chat UIs.
 */
class YALMRClient {
    #baseUrl;
    #sessionId = null;
    #model = null;

    constructor(baseUrl = '') {
        this.#baseUrl = baseUrl.replace(/\/$/, '');
    }

    get sessionId() { return this.#sessionId; }
    get model() { return this.#model; }

    /** Returns the list of loaded models from /v1/models. */
    async models() {
        const res = await fetch(`${this.#baseUrl}/v1/models`);
        if (!res.ok) throw new Error(`models: HTTP ${res.status}`);
        return (await res.json()).data ?? [];
    }

    /** Creates a new persistent session. Must be called before send(). */
    async createSession(model, system = null) {
        const body = { model };
        if (system) body.system = system;
        const res = await fetch(`${this.#baseUrl}/v1/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!res.ok) throw new Error(`createSession: HTTP ${res.status}`);
        const data = await res.json();
        this.#sessionId = data.id;
        this.#model = data.model;
        return data.id;
    }

    /**
     * Sends a user message to the active session.
     * @param {string} message
     * @param {{ stream?: boolean, images?: string[], onToken?: (delta: string, full: string) => void, onEnd?: (event: object) => void }} options
     * @returns {Promise<string>} The complete assistant reply.
     */
    async send(message, { stream = true, images = [], enableThinking = null, onToken = null, onThinking = null, onToolCall = null, onToolResult = null, onEnd = null } = {}) {
        if (!this.#sessionId) throw new Error('No active session — call createSession() first.');

        let bodyObj;
        if (images && images.length > 0) {
            const parts = images.map(url => ({ type: 'image_url', image_url: { url } }));
            if (message) parts.push({ type: 'text', text: message });
            bodyObj = { parts, stream };
        } else {
            bodyObj = { message, stream };
        }
        if (enableThinking !== null) bodyObj.enable_thinking = enableThinking;

        const res = await fetch(`${this.#baseUrl}/v1/sessions/${this.#sessionId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(bodyObj)
        });
        if (!res.ok) throw new Error(`send: HTTP ${res.status}`);

        if (!stream) {
            const data = await res.json();
            const content = data.message?.content;
            return typeof content === 'string' ? content : (content?.text ?? '');
        }

        return this.#consumeSse(res, { onToken, onThinking, onToolCall, onToolResult, onEnd });
    }

    /** Deletes the active session on the server and clears local state. */
    async deleteSession() {
        if (!this.#sessionId) return;
        await fetch(`${this.#baseUrl}/v1/sessions/${this.#sessionId}`, { method: 'DELETE' });
        this.#sessionId = null;
        this.#model = null;
    }

    async #consumeSse(res, { onToken, onThinking, onToolCall, onToolResult, onEnd } = {}) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let full = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const blocks = buffer.split('\n\n');
            buffer = blocks.pop();
            for (const block of blocks) {
                const { event, data } = YALMRClient.#parseBlock(block);
                if (event === 'token' && data?.delta) {
                    full += data.delta;
                    onToken?.(data.delta, full);
                } else if (event === 'thinking' && data?.delta) {
                    onThinking?.(data.delta);
                } else if (event === 'tool_call' && data) {
                    onToolCall?.(data);
                } else if (event === 'tool_result' && data) {
                    onToolResult?.(data);
                } else if (event === 'end') {
                    onEnd?.(data);
                }
            }
        }

        return full;
    }

    static #parseBlock(block) {
        let event = null, data = null;
        for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) event = line.slice(7).trim();
            else if (line.startsWith('data: ')) {
                try { data = JSON.parse(line.slice(6)); } catch { data = line.slice(6); }
            }
        }
        return { event, data };
    }
}

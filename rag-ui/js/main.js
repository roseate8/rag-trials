// Global state
let currentQuery = '';
let isProcessing = false;
let conversationHistory = [];

// DOM Elements
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const chunkTypeSelect = document.getElementById('chunkTypeSelect');
const searchBtn = document.getElementById('searchBtn');
const historyBtn = document.getElementById('historyBtn');
const answerContent = document.getElementById('answerContent');
const thinkingSection = document.getElementById('thinkingSection');
const logsContent = document.getElementById('logsContent');
const metricsContent = document.getElementById('metricsContent');
const sourcesContent = document.getElementById('sourcesContent');
const historyPanel = document.getElementById('historyPanel');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const chatsList = document.getElementById('chatsList');
const chatDetails = document.getElementById('chatDetails');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadConversationHistory();
});

// Event Listeners
function initializeEventListeners() {
    queryForm.addEventListener('submit', handleQuerySubmit);
    historyBtn.addEventListener('click', toggleHistoryPanel);
    closeHistoryBtn.addEventListener('click', closeHistoryPanel);
    
    // Close history panel when clicking outside
    document.addEventListener('click', function(e) {
        if (historyPanel.classList.contains('open') && 
            !historyPanel.contains(e.target) && 
            !historyBtn.contains(e.target)) {
            closeHistoryPanel();
        }
    });
}

// Handle query submission
async function handleQuerySubmit(e) {
    e.preventDefault();
    
    if (isProcessing) return;
    
    const query = queryInput.value.trim();
    const chunkType = chunkTypeSelect.value;
    if (!query) {
        showError('Please enter a query');
        return;
    }
    
    currentQuery = query;
    startProcessing();
    
    try {
        await processQuery(query, chunkType);
    } catch (error) {
        handleError(error);
    } finally {
        stopProcessing();
    }
}

// Start processing state
function startProcessing() {
    isProcessing = true;
    searchBtn.disabled = true;
    searchBtn.textContent = 'Processing...';
    queryInput.disabled = true;
    
    // Clear previous results
    clearResults();
    
    // Show thinking section
    thinkingSection.style.display = 'block';
    thinkingSection.open = true;
    
    // Show initial state
    answerContent.innerHTML = '<div class="thinking-indicator">Thinking...</div>';
    
    // Clear logs
    logsContent.textContent = '';
}

// Stop processing state
function stopProcessing() {
    isProcessing = false;
    searchBtn.disabled = false;
    searchBtn.textContent = 'Search';
    queryInput.disabled = false;
    queryInput.value = '';
}

// Clear previous results
function clearResults() {
    answerContent.innerHTML = '';
    resetMetrics();
    sourcesContent.innerHTML = '<div class="empty-state">No sources available</div>';
}

// Process query and stream results
async function processQuery(query, chunkType = 'layout-aware') {
    try {
        // Start streaming logs
        const logStream = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query, chunk_type: chunkType })
        });

        if (!logStream.ok) {
            throw new Error(`HTTP error! status: ${logStream.status}`);
        }

        const reader = logStream.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let result = null;

        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.trim()) {
                    // Handle Server-Sent Events format (data: {...})
                    if (line.startsWith('data: ')) {
                        try {
                            const jsonStr = line.substring(6); // Remove 'data: ' prefix
                            const data = JSON.parse(jsonStr);
                            
                            if (data.type === 'log') {
                                appendLog(data.message);
                            } else if (data.type === 'result') {
                                result = data.data;
                            } else if (data.type === 'error') {
                                throw new Error(data.message);
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                            appendLog(`Parse error: ${line}`);
                        }
                    } else if (line.trim() !== '') {
                        // If not SSE format, treat as regular log
                        appendLog(line);
                    }
                }
            }
        }

        if (result) {
            displayResults(result);
            saveToHistory(query, result);
        } else {
            throw new Error('No result received');
        }

    } catch (error) {
        throw error;
    }
}

// Append log message
function appendLog(message) {
    // Add timestamp only if message doesn't already have one
    const finalMessage = message.includes('[') && message.includes(']') ? 
        message : `[${new Date().toLocaleTimeString()}] ${message}`;
    
    logsContent.textContent += `${finalMessage}\n`;
    logsContent.scrollTop = logsContent.scrollHeight;
}

// Display query results
function displayResults(result) {
    const { llm_response, timing, resources, retrieved_chunks } = result;
    
    // Display answer with the query at the top
    const q = llm_response.query || '';
    const a = llm_response.answer || '';
    answerContent.textContent = (q ? `Q: ${q}\n\n` : '') + a;
    
    // Display metrics
    displayMetrics(timing, resources, llm_response.tokens_used);
    
    // Display sources
    displaySources(retrieved_chunks);
}

// Display metrics
function displayMetrics(timing, resources, tokens) {
    document.getElementById('totalTime').textContent = timing ? `${timing.total_time.toFixed(2)}s` : '--';
    document.getElementById('tokensUsed').textContent = tokens || '--';
    
    if (resources && resources.peak_cpu_percent !== undefined) {
        document.getElementById('cpuPeak').textContent = `${resources.peak_cpu_percent.toFixed(1)}%`;
    } else {
        document.getElementById('cpuPeak').textContent = '--';
    }
    
    if (resources && resources.peak_memory_mb !== undefined) {
        document.getElementById('memoryPeak').textContent = `${resources.peak_memory_mb.toFixed(1)} MB`;
    } else {
        document.getElementById('memoryPeak').textContent = '--';
    }
}

// Reset metrics
function resetMetrics() {
    document.getElementById('totalTime').textContent = '--';
    document.getElementById('tokensUsed').textContent = '--';
    document.getElementById('cpuPeak').textContent = '--';
    document.getElementById('memoryPeak').textContent = '--';
}

// Sigmoid helper to convert logits to probabilities
function sigmoid(x) {
    if (typeof x !== 'number' || !isFinite(x)) return null;
    if (x > 20) return 1;       // avoid overflow
    if (x < -20) return 0;      // avoid underflow
    return 1 / (1 + Math.exp(-x));
}

// Format probabilities: show 3 decimals normally, scientific notation when tiny
function formatProb(p) {
    if (p === null || typeof p !== 'number' || !isFinite(p)) return '--';
    if (p > 0 && p < 0.001) return p.toExponential(2); // e.g., 3.05e-5
    if (p > 0.999) return '0.999+';
    if (p < 0.001) return '0.000-';
    return p.toFixed(3);
}

// Toggle full text display (for main sources)
function toggleFullText(index) {
    toggleFullTextWithPrefix('', index);
}

// Toggle full text display with prefix support (for both main and history)
function toggleFullTextWithPrefix(prefix, index) {
    const chunkId = `${prefix}snippet-${index}`;
    const snippetElement = document.getElementById(chunkId);
    
    if (!snippetElement) {
        console.error(`Element not found: ${chunkId}`);
        return;
    }
    
    const isExpanded = snippetElement.classList.contains('expanded');
    const fullText = decodeURIComponent(snippetElement.dataset.fullText);
    const truncatedText = decodeURIComponent(snippetElement.dataset.truncated);
    
    if (isExpanded) {
        // Collapse - show truncated version
        snippetElement.innerHTML = `
            ${truncatedText}
            <button class="see-more-btn" onclick="toggleFullTextWithPrefix('${prefix}', ${index})">
                see more
            </button>
        `;
        snippetElement.classList.remove('expanded');
    } else {
        // Expand - show full text
        snippetElement.innerHTML = `
            ${fullText}
            <button class="see-more-btn" onclick="toggleFullTextWithPrefix('${prefix}', ${index})">
                see less
            </button>
        `;
        snippetElement.classList.add('expanded');
    }
}

// History panel functions
function toggleHistoryPanel() {
    historyPanel.classList.toggle('open');
    if (historyPanel.classList.contains('open')) {
        loadConversationHistory();
    }
}

function closeHistoryPanel() {
    historyPanel.classList.remove('open');
}

// Load conversation history
async function loadConversationHistory() {
    try {
        const response = await fetch('/api/history');
        if (response.ok) {
            conversationHistory = await response.json();
            displayConversationHistory();
        } else {
            chatsList.innerHTML = '<div class="empty-state">No saved chats</div>';
        }
    } catch (error) {
        console.error('Failed to load conversation history:', error);
        chatsList.innerHTML = '<div class="empty-state">Failed to load history</div>';
    }
}

// Display conversation history
function displayConversationHistory() {
    if (!conversationHistory || conversationHistory.length === 0) {
        chatsList.innerHTML = '<div class="empty-state">No saved chats</div>';
        return;
    }
    
    const historyHtml = conversationHistory
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        .map((chat, index) => {
            const timestamp = new Date(chat.timestamp).toLocaleString();
            const query = chat.query.length > 60 ? chat.query.substring(0, 60) + '...' : chat.query;
            
            return `
                <div class="chat-item" onclick="selectChat(${index})" data-index="${index}">
                    <div class="chat-timestamp">${timestamp}</div>
                    <div class="chat-query">${query}</div>
                </div>
            `;
        }).join('');
    
    chatsList.innerHTML = historyHtml;
}

// Select chat from history
function selectChat(index) {
    // Remove previous selection
    document.querySelectorAll('.chat-item').forEach(item => item.classList.remove('selected'));
    
    // Add selection to clicked item
    document.querySelector(`[data-index="${index}"]`).classList.add('selected');
    
    const chat = conversationHistory[index];
    displayChatDetails(chat);
}

// Generate chunk HTML (shared between sources and history)
function generateChunkHtml(chunks, prefix = '') {
    if (!chunks || chunks.length === 0) {
        return '<div class="empty-state">No sources</div>';
    }
    
    return chunks.map((chunk, index) => {
        const payload = chunk.payload || {};
        const fullText = payload.text || 'No content';
        
        // Determine chunk type - use actual data, default to blank
        let chunkType = payload.chunk_type || '';
        
        // Infer chunk type from available metadata if not explicitly set
        if (!chunkType && payload.metadata) {
            const meta = payload.metadata;
            if (meta.json_table_id) {
                chunkType = 'table';
            } else if (meta.json_label) {
                chunkType = meta.json_label; // Could be 'text', 'footnote', 'image', etc.
            }
        }
        
        // Final fallback: check text content for table indicators
        if (!chunkType && fullText.startsWith('Table: ')) {
            chunkType = 'table';
        }
        // Otherwise leave blank if no indicators found
        const isTextTruncated = fullText.length > 400;
        const snippet = isTextTruncated ? fullText.substring(0, 400) + '...' : fullText;
        
        const scores = [];
        if (typeof chunk.score === 'number') {
            scores.push(`similarity: ${chunk.score.toFixed(4)}`);
        }
        if (typeof chunk.rerank_score === 'number' && isFinite(chunk.rerank_score)) {
            const prob = sigmoid(chunk.rerank_score);
            scores.push(`rerank (logit): ${chunk.rerank_score.toFixed(4)}`);
            if (prob !== null) {
                scores.push(`rerank (prob): ${formatProb(prob)}`);
            }
        }
        
        // Collect ALL metadata fields as tags
        const metadataTags = [];
        
        // Core fields
        if (payload.method) metadataTags.push(payload.method);
        if (payload.chunk_type) metadataTags.push(payload.chunk_type);
        if (payload.document) metadataTags.push(payload.document);
        
        // Location/structure fields
        if (payload.section_title) metadataTags.push(payload.section_title);
        if (payload.page_number) metadataTags.push(`page ${payload.page_number}`);
        if (payload.start_char !== undefined && payload.end_char !== undefined) {
            metadataTags.push(`chars ${payload.start_char}-${payload.end_char}`);
        }
        
        // Chunk metadata
        if (payload.chunk_id !== undefined) metadataTags.push(`chunk ${payload.chunk_id}`);
        if (payload.chunk_size) metadataTags.push(`size ${payload.chunk_size}`);
        if (payload.overlap) metadataTags.push(`overlap ${payload.overlap}`);
        
        // JSON-specific metadata
        if (payload.metadata) {
            const meta = payload.metadata;
            if (meta.page_no && meta.page_no !== payload.page_number) metadataTags.push(`page ${meta.page_no}`);
            if (meta.json_label) metadataTags.push(meta.json_label);
            if (meta.json_table_id) metadataTags.push(meta.json_table_id);
            if (meta.bbox) {
                metadataTags.push('positioned');
            }
        }
        
        const chunkId = `${prefix}snippet-${index}`;
        
        return `
            <div class="source-chunk">
                <div class="source-header">
                    ${chunkType ? `<span class="chunk-type">${chunkType}</span>` : ''}
                </div>
                <div class="source-snippet" id="${chunkId}" data-full-text="${encodeURIComponent(fullText)}" data-truncated="${encodeURIComponent(snippet)}">
                    ${snippet}
                    ${isTextTruncated ? `
                        <button class="see-more-btn" onclick="toggleFullTextWithPrefix('${prefix}', ${index})">
                            see more
                        </button>
                    ` : ''}
                </div>
                ${scores.length > 0 ? `
                    <div class="source-scores">
                        ${scores.map(score => `<span class="score">${score}</span>`).join('')}
                    </div>
                ` : ''}
                ${metadataTags.length > 0 ? `
                    <div class="metadata-tags">
                        ${metadataTags.map(tag => `<span class="metadata-tag">${tag}</span>`).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// Display sources (updated to use shared function)
function displaySources(chunks) {
    if (!chunks || chunks.length === 0) {
        sourcesContent.innerHTML = '<div class="empty-state">No sources found</div>';
        return;
    }
    
    sourcesContent.innerHTML = generateChunkHtml(chunks);
}

// Display chat details
function displayChatDetails(chat) {
    const result = chat.result;
    const chunks = result.retrieved_chunks || [];
    
    const chunksHtml = generateChunkHtml(chunks, 'history-');
    
    chatDetails.innerHTML = `
        <div class="detail-section">
            <h4>Q: Query</h4>
            <div class="detail-content">${chat.query}</div>
        </div>
        <div class="detail-section">
            <h4>A: Answer</h4>
            <div class="detail-content">${result.llm_response.answer}</div>
        </div>
        <div class="detail-section">
            <h4>Top Chunks</h4>
            <div class="detail-content">${chunksHtml}</div>
        </div>
    `;
}

// Save to history
function saveToHistory(query, result) {
    const historyEntry = {
        timestamp: new Date().toISOString(),
        query: query,
        result: result,
        tokens_used: result.llm_response.tokens_used,
        timing: result.timing || {},
        resources: result.resources || {}
    };
    
    // Add to local history
    conversationHistory.unshift(historyEntry);
    
    // Optionally save to server (if endpoint exists)
    fetch('/api/history', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(historyEntry)
    }).catch(error => console.error('Failed to save to server history:', error));
}

// Error handling
function handleError(error) {
    console.error('Query processing error:', error);
    answerContent.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
    appendLog(`ERROR: ${error.message}`);
}

function showError(message) {
    // Simple error display - could be enhanced with a proper notification system
    alert(message);
}

// Make functions globally available
window.selectChat = selectChat;
window.toggleFullText = toggleFullText;
window.toggleFullTextWithPrefix = toggleFullTextWithPrefix;
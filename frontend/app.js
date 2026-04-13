const API_URL = 'http://localhost:8000/api';

// DOM Elements
const chatBox = document.getElementById('chatBox');
const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const modeSelect = document.getElementById('modeSelect');
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadStatus = document.getElementById('uploadStatus');
const statsGrid = document.getElementById('statsGrid');
const toolsList = document.getElementById('toolsList');
const modal = document.getElementById('detailsModal');
const modalBody = document.getElementById('modalBody');
const modalTitle = document.getElementById('modalTitle');
const closeModalBtn = document.getElementById('closeModal');
const manualBtn = document.getElementById('manualBtn');
const manualPanel = document.getElementById('manualPanel');
const closeManualBtn = document.getElementById('closeManualBtn');

// Configure Marked.js
marked.setOptions({
    breaks: true,
    gfm: true
});

// Auto-resize textarea
questionInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

questionInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// Initialize on load
async function init() {
    await fetchStats();
    await fetchTools();
}

async function fetchTools() {
    try {
        const response = await fetch(`${API_URL}/tools`);
        if (!response.ok) throw new Error('Failed to fetch tools');
        const data = await response.json();
        renderTools(data.tools);
    } catch (error) {
        console.error(error);
        if (toolsList) {
            toolsList.innerHTML = `<div style="text-align: center; color: var(--error)">Error loading tools</div>`;
        }
    }
}

function renderTools(tools) {
    if (!toolsList) return;
    toolsList.innerHTML = '';
    
    tools.forEach(tool => {
        const card = document.createElement('div');
        card.className = 'tool-card';
        
        let inputsHtml = '';
        const schema = tool.input_schema || {};
        for (const [key, typeStr] of Object.entries(schema)) {
            const isOptional = typeStr.includes('optional');
            const placeholder = isOptional ? '(Optional)' : '(Required)';
            inputsHtml += `
                <div class="tool-form-group">
                    <label>${key} <span style="text-transform:lowercase; color:var(--text-muted)">- ${typeStr}</span></label>
                    <input type="text" class="tool-input-field" id="tool-input-${tool.name}-${key}" placeholder="${placeholder}" ${isOptional ? '' : 'required'}>
                </div>
            `;
        }
        
        card.innerHTML = `
            <div class="tool-header" onclick="this.nextElementSibling.classList.toggle('active')">
                <div class="tool-name">
                    <i class="ri-tools-line"></i> ${tool.name}
                </div>
                <i class="ri-arrow-down-s-line" style="color:var(--text-muted)"></i>
            </div>
            <div class="tool-body">
                <div class="tool-desc">${tool.description}</div>
                <form onsubmit="handleToolSubmit(event, '${tool.name}')">
                    ${inputsHtml}
                    <button type="submit" class="tool-submit-btn" id="btn-submit-${tool.name}">
                        <i class="ri-play-line"></i> Execute
                    </button>
                </form>
            </div>
        `;
        toolsList.appendChild(card);
    });
}

window.handleToolSubmit = async function(e, toolName) {
    e.preventDefault();
    
    const form = e.target;
    const inputs = form.querySelectorAll('.tool-input-field');
    const payload = {};
    let hasError = false;
    
    inputs.forEach(input => {
        const key = input.id.replace(`tool-input-${toolName}-`, '');
        const val = input.value.trim();
        if (input.required && !val) {
            hasError = true;
        }
        if (val !== '') {
            payload[key] = val;
        }
    });
    
    if (hasError) return;
    
    const btn = document.getElementById(`btn-submit-${toolName}`);
    btn.disabled = true;
    btn.innerHTML = `<i class="ri-loader-4-line ri-spin"></i> Executing...`;
    
    appendMessage('user', `*Manually executing tool **\`${toolName}\`** with payload:* \n\`\`\`json\n${JSON.stringify(payload, null, 2)}\n\`\`\``);
    const typingElement = showTyping();
    
    try {
        const response = await fetch(`${API_URL}/tool`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: toolName, input: payload })
        });
        
        chatBox.removeChild(typingElement);
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || "Tool execution failed");
        }
        
        appendMessage('assistant', `**Tool Result (${toolName}):**\n\n${result.observation || 'Success'}`);
        
    } catch (error) {
        if(chatBox.contains(typingElement)) chatBox.removeChild(typingElement);
        appendMessage('assistant', `**Tool Error:** ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="ri-play-line"></i> Execute`;
        form.reset();
    }
}

async function fetchStats() {
    try {
        const response = await fetch(`${API_URL}/stats`);
        if (!response.ok) throw new Error('Failed to fetch stats');
        const stats = await response.json();
        
        statsGrid.innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${stats.chunks_indexed || 0}</div>
                <div class="stat-label">Chunks Indexed</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.tools_available || 0}</div>
                <div class="stat-label">Tools</div>
            </div>
            <div class="stat-item" style="grid-column: span 2;">
                <div class="stat-value text-gradient">${settingsMap(stats.domain)}</div>
                <div class="stat-label">Domain</div>
            </div>
        `;
    } catch (error) {
        console.error(error);
        statsGrid.innerHTML = `<div style="grid-column: span 2; text-align: center; color: var(--error)">Error loading stats API down?</div>`;
    }
}

function settingsMap(domain) {
    if (!domain || domain === "auto") return "Auto";
    return domain.charAt(0).toUpperCase() + domain.slice(1);
}

// Upload Handling
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--primary)';
});
uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--border-glass-light)';
});
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--border-glass-light)';
    if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileUpload(e.target.files[0]);
    }
});

async function handleFileUpload(file) {
    uploadStatus.textContent = "Uploading: " + file.name + "...";
    uploadStatus.className = "status-msg";
    uploadProgress.classList.add('active');
    
    // Animate fake progress
    const bar = uploadProgress.querySelector('.progress-bar');
    bar.style.width = '30%';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/ingest`, {
            method: 'POST',
            body: formData
        });
        
        bar.style.width = '90%';
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Ingestion failed");
        }
        
        bar.style.width = '100%';
        uploadStatus.textContent = `Success! Inherited ${data.child_chunks} chunks.`;
        uploadStatus.className = "status-msg success";
        
        // Refresh stats
        setTimeout(() => {
            fetchStats();
            uploadProgress.classList.remove('active');
            bar.style.width = '0%';
        }, 1500);

    } catch (error) {
        bar.style.width = '100%';
        bar.style.background = 'var(--error)';
        uploadStatus.textContent = error.message;
        uploadStatus.className = "status-msg error";
        setTimeout(() => {
            uploadProgress.classList.remove('active');
            bar.style.width = '0%';
            bar.style.background = 'var(--primary-gradient)';
        }, 3000);
    }
    
    fileInput.value = ''; // reset
}

// Chat Handling
function clearChat() {
    chatBox.innerHTML = `
        <div class="message assistant slide-in-bottom">
            <div class="avatar"><i class="ri-robot-2-line"></i></div>
            <div class="msg-bubble">
                <div class="msg-content markdown-body">
                    <p>Chat cleared. Ready for a new query.</p>
                </div>
            </div>
        </div>
    `;
}

function appendMessage(role, content, rawHTML = false) {
    const isUser = role === 'user';
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user' : 'assistant'} slide-in-bottom`;
    
    const icon = isUser ? '<i class="ri-user-line"></i>' : '<i class="ri-robot-2-line"></i>';
    
    msgDiv.innerHTML = `
        <div class="avatar">${icon}</div>
        <div class="msg-bubble">
            <div class="msg-content markdown-body">
                ${rawHTML ? content : marked.parse(content)}
            </div>
        </div>
    `;
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

function showTyping() {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message assistant typing-msg slide-in-bottom`;
    msgDiv.innerHTML = `
        <div class="avatar"><i class="ri-robot-2-line"></i></div>
        <div class="msg-bubble">
            <div class="typing-indicator">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
        </div>
    `;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    // Reset input
    questionInput.value = '';
    questionInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Append user message
    appendMessage('user', question);

    // Show typing
    const typingElement = showTyping();

    try {
        const mode = modeSelect.value;
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, mode })
        });
        
        chatBox.removeChild(typingElement);

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Error calling API");
        }

        // Add main answer
        const msgDiv = appendMessage('assistant', data.answer);
        
        // Add metadata beneath answer
        const bubble = msgDiv.querySelector('.msg-bubble');
        const metaDiv = document.createElement('div');
        metaDiv.className = 'msg-meta';
        
        // Mode Badge
        metaDiv.innerHTML += `<span class="badge-btn" style="cursor:default; background: rgba(59,130,246,0.1); color: var(--primary)">
            <i class="ri-route-line"></i> Route: ${data.mode || mode}
        </span>`;

        // If direct RAG, sources
        if (data.sources_used && data.sources_used.length > 0) {
            const btn = document.createElement('button');
            btn.className = 'badge-btn';
            btn.innerHTML = `<i class="ri-book-read-line"></i> Sources (${data.sources_used.length})`;
            btn.onclick = () => showSources(data.sources_used);
            metaDiv.appendChild(btn);
        }

        // If Agent, trace
        if (data.steps && data.steps.length > 0) {
            const btn = document.createElement('button');
            btn.className = 'badge-btn';
            btn.innerHTML = `<i class="ri-node-tree"></i> View Agent Trace (${data.steps.length} steps)`;
            btn.onclick = () => showTrace(data.steps, data.route_reason);
            metaDiv.appendChild(btn);
        }

        bubble.appendChild(metaDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (error) {
        if(chatBox.contains(typingElement)) {
            chatBox.removeChild(typingElement);
        }
        appendMessage('assistant', `**Error:** ${error.message}`);
    } finally {
        sendBtn.disabled = false;
        questionInput.focus();
    }
});

// Modals
closeModalBtn.addEventListener('click', () => modal.classList.remove('active'));
modal.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-backdrop')) {
        modal.classList.remove('active');
    }
});

function showSources(sources) {
    modalTitle.innerHTML = '<i class="ri-book-read-line"></i> Retrieved Sources';
    let html = '';
    sources.forEach(s => {
        html += `
            <div class="source-item">
                <h4>${s.file} (Page ${s.page})</h4>
                <p><strong>Section:</strong> ${s.section}</p>
                <div class="markdown-body" style="margin-top: 0.5rem; font-size: 0.85rem">
                    ${marked.parse("> " + (s.text ? s.text.substring(0, 300) + '...' : 'Preview not available'))}
                </div>
            </div>
        `;
    });
    modalBody.innerHTML = html;
    modal.classList.add('active');
}

function showTrace(steps, reason) {
    modalTitle.innerHTML = '<i class="ri-node-tree"></i> Agentic Reasoning Trace';
    let html = reason ? `<p style="margin-bottom: 1.5rem; color: var(--text-muted)"><em>Routing Reason: ${reason}</em></p>` : '';
    
    steps.forEach((step, idx) => {
        html += `
            <div class="trace-step">
                <div class="stone">Step ${idx + 1}</div>
                <div class="trace-content" style="margin-bottom: 0.5rem">
                    <strong>Thought:</strong> ${step.thought}
                </div>
                ${step.tool_name ? `
                    <div style="font-size: 0.85rem; margin-bottom: 0.5rem; color: var(--warning)">
                        <i class="ri-tools-fill"></i> Calling <strong>${step.tool_name}</strong>
                    </div>
                ` : ''}
                ${step.observation ? `
                    <div class="trace-content">
                        <strong>Observation:</strong><br> ${step.observation.substring(0, 500)}...
                    </div>
                ` : ''}
            </div>
        `;
    });
    modalBody.innerHTML = html;
    modal.classList.add('active');
}

// Manual Handling
if (manualBtn && manualPanel && closeManualBtn) {
    manualBtn.addEventListener('click', () => {
        manualPanel.classList.add('active');
    });
    closeManualBtn.addEventListener('click', () => {
        manualPanel.classList.remove('active');
    });
}

// Start
window.addEventListener('DOMContentLoaded', init);

// Flying Emojis Animation
const emojiIcons = ['📚', '🧠', '🔍', '💡', '⚛️', '✨', '📖', '🚀'];
function createFlyingEmoji() {
    const glowBg = document.querySelector('.glow-bg');
    if (!glowBg) return;

    const el = document.createElement('div');
    el.className = 'flying-emoji';
    el.textContent = emojiIcons[Math.floor(Math.random() * emojiIcons.length)];

    const size = Math.random() * 2 + 1; 
    el.style.fontSize = `${size}rem`;
    el.style.left = `${Math.random() * 100}vw`;

    const duration = Math.random() * 20 + 20; 
    el.style.animationDuration = `${duration}s`;
    
    // Slight random drift offsets
    const xDrift = (Math.random() * 100 - 50) + 'px';
    el.style.setProperty('--x-drift', xDrift);

    el.style.opacity = `${Math.random() * 0.3 + 0.2}`; // Opacity between 0.2 and 0.5
    el.style.filter = `blur(${Math.random() * 1.5}px)`; // Blur between 0px and 1.5px

    glowBg.appendChild(el);

    setTimeout(() => {
        if (glowBg.contains(el)) el.remove();
    }, duration * 1000);
}

setInterval(createFlyingEmoji, 2500);

for(let i=0; i<6; i++) {
    setTimeout(() => {
        const el = createFlyingEmoji();
        // Give them a head start by dropping them lower immediately?
        // Timeout creates staggered spawn
    }, Math.random() * 5000);
}

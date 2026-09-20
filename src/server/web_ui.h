#pragma once

namespace murdok {

inline const char* get_embedded_web_ui() {
    return R"rawhtml(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MuRDoK Inference Engine</title>
    <style>
        :root {
            --bg-base: #0c0f14;
            --bg-card: #151922;
            --bg-bubble: #1c2230;
            --bg-user: #1e3a5f;
            --border-color: #273142;
            --text-main: #f0f4fc;
            --text-dim: #8b9bb4;
            --accent: #38bdf8;
            --accent-green: #34d399;
            --accent-purple: #c084fc;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background-color: var(--bg-base); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        header {
            background-color: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand { display: flex; align-items: center; gap: 10px; }
        .logo { font-weight: 800; font-size: 1.25rem; letter-spacing: 0.05em; color: var(--accent); }
        .tagline { font-size: 0.75rem; color: var(--text-dim); background: var(--bg-bubble); padding: 3px 8px; border-radius: 6px; border: 1px solid var(--border-color); }
        .hud-stats { display: flex; gap: 14px; font-size: 0.8rem; }
        .hud-pill { background: var(--bg-bubble); padding: 5px 10px; border-radius: 6px; border: 1px solid var(--border-color); display: flex; gap: 6px; }
        .hud-pill span { color: var(--text-dim); }
        .hud-pill strong { color: var(--accent-green); }

        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
        }
        .msg {
            display: flex;
            flex-direction: column;
            max-width: 80%;
            padding: 14px 18px;
            border-radius: 12px;
            line-height: 1.5;
            font-size: 0.95rem;
            word-break: break-word;
            white-space: pre-wrap;
        }
        .msg.user { align-self: flex-end; background-color: var(--bg-user); border: 1px solid #2b5282; }
        .msg.assistant { align-self: flex-start; background-color: var(--bg-bubble); border: 1px solid var(--border-color); }
        .msg-meta { font-size: 0.7rem; color: var(--text-dim); margin-bottom: 4px; font-weight: 600; text-transform: uppercase; }
        .msg-stats { font-size: 0.72rem; color: var(--accent-purple); margin-top: 8px; border-top: 1px solid var(--border-color); padding-top: 4px; }

        footer {
            background-color: var(--bg-card);
            border-top: 1px solid var(--border-color);
            padding: 16px 24px;
            display: flex;
            justify-content: center;
        }
        .input-box {
            display: flex;
            gap: 10px;
            max-width: 900px;
            width: 100%;
        }
        textarea {
            flex: 1;
            background: var(--bg-base);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 0.95rem;
            resize: none;
            height: 52px;
            outline: none;
        }
        textarea:focus { border-color: var(--accent); }
        button {
            background: var(--accent);
            color: #000;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0 22px;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="logo">MuRDoK</div>
            <div class="tagline">Local Inference Runtime</div>
        </div>
        <div class="hud-stats">
            <div class="hud-pill"><span>Model:</span> <strong id="hud-model">Loading...</strong></div>
            <div class="hud-pill"><span>Throughput:</span> <strong id="hud-tps">-- tok/s</strong></div>
            <div class="hud-pill"><span>TTFT:</span> <strong id="hud-ttft">-- ms</strong></div>
            <div class="hud-pill"><span>RAM:</span> <strong id="hud-ram">-- MB</strong></div>
        </div>
    </header>

    <main id="chat-container">
        <div class="msg assistant">
            <div class="msg-meta">MuRDoK Engine</div>
            <div>Welcome! MuRDoK Inference Engine is active and ready on this local machine. Enter a prompt below to begin.</div>
        </div>
    </main>

    <footer>
        <div class="input-box">
            <textarea id="prompt-input" placeholder="Type a message or instruction (Press Enter to send)..." rows="1"></textarea>
            <button id="send-btn">Send</button>
        </div>
    </footer>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');

        // Fetch model info on startup
        fetch('/v1/models')
            .then(r => r.json())
            .then(d => {
                if (d.data && d.data.length > 0) {
                    document.getElementById('hud-model').textContent = d.data[0].id;
                }
            }).catch(e => console.error(e));

        function appendMessage(role, text, stats = null) {
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            const meta = document.createElement('div');
            meta.className = 'msg-meta';
            meta.textContent = role === 'user' ? 'You' : 'MuRDoK';
            div.appendChild(meta);

            const content = document.createElement('div');
            content.className = 'msg-content';
            content.textContent = text;
            div.appendChild(content);

            if (stats) {
                const statsDiv = document.createElement('div');
                statsDiv.className = 'msg-stats';
                statsDiv.textContent = stats;
                div.appendChild(statsDiv);
            }

            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return content;
        }

        async function sendPrompt() {
            const text = promptInput.value.trim();
            if (!text) return;
            promptInput.value = '';
            appendMessage('user', text);
            sendBtn.disabled = true;

            const assistantMsgContent = appendMessage('assistant', 'Thinking...');
            assistantMsgContent.textContent = '';

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: [{ role: 'user', content: text }],
                        stream: true
                    })
                });

                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let fullText = '';
                let statsText = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.replace('data: ', '').trim();
                            if (dataStr === '[DONE]') break;
                            try {
                                const parsed = JSON.parse(dataStr);
                                if (parsed.choices && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                                    fullText += parsed.choices[0].delta.content;
                                    assistantMsgContent.textContent = fullText;
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                                if (parsed.metrics) {
                                    const m = parsed.metrics;
                                    document.getElementById('hud-tps').textContent = `${m.gen_tok_per_sec.toFixed(1)} tok/s`;
                                    document.getElementById('hud-ttft').textContent = `${m.ttft_ms.toFixed(0)} ms`;
                                    document.getElementById('hud-ram').textContent = `${m.peak_ram_mb.toFixed(0)} MB`;
                                    statsText = `${m.gen_tok_per_sec.toFixed(1)} tok/s | TTFT: ${m.ttft_ms.toFixed(0)} ms | TPOT: ${m.tpot_ms.toFixed(1)} ms | RAM: ${m.peak_ram_mb.toFixed(0)} MB`;
                                }
                            } catch (e) {}
                        }
                    }
                }
                if (statsText) {
                    const statsDiv = document.createElement('div');
                    statsDiv.className = 'msg-stats';
                    statsDiv.textContent = statsText;
                    assistantMsgContent.parentElement.appendChild(statsDiv);
                }
            } catch (err) {
                assistantMsgContent.textContent = "Error communicating with MuRDoK engine: " + err;
            } finally {
                sendBtn.disabled = false;
                promptInput.focus();
            }
        }

        sendBtn.addEventListener('click', sendPrompt);
        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendPrompt();
            }
        });
    </script>
</body>
</html>
)rawhtml";
}

} // namespace murdok

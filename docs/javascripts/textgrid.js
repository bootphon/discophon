// ── TextGrid Parser ──────────────────────────────────────────────
function parseTextGrid(text) {
    text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    if (text.includes('Object class = "TextGrid"')) {
        return parseTextGridNormal(text);
    }
    return parseTextGridShort(text);
}

function parseTextGridNormal(text) {
    const tiers = [];
    const tierBlocks = text.split(/item\s*\[\d+\]/).slice(1);
    for (const block of tierBlocks) {
        const tierClass = (block.match(/class\s*=\s*"(\w+)"/) || [])[1];
        const tierName = (block.match(/name\s*=\s*"([^"]*)"/) || [])[1] || '';
        const tier = { name: tierName, type: tierClass, items: [] };
        if (tierClass === 'IntervalTier') {
            const intervals = block.split(/intervals\s*\[\d+\]/).slice(1);
            for (const iv of intervals) {
                const xmin = parseFloat((iv.match(/xmin\s*=\s*([\d.eE+-]+)/) || [])[1]);
                const xmax = parseFloat((iv.match(/xmax\s*=\s*([\d.eE+-]+)/) || [])[1]);
                const label = (iv.match(/text\s*=\s*"([^"]*)"/) || [])[1] || '';
                tier.items.push({ xmin, xmax, text: label });
            }
        } else if (tierClass === 'TextTier') {
            const points = block.split(/points\s*\[\d+\]/).slice(1);
            for (const pt of points) {
                const time = parseFloat((pt.match(/(?:number|time)\s*=\s*([\d.eE+-]+)/) || [])[1]);
                const mark = (pt.match(/(?:mark|value)\s*=\s*"([^"]*)"/) || [])[1] || '';
                tier.items.push({ time, text: mark });
            }
        }
        tiers.push(tier);
    }
    return tiers;
}

function parseTextGridShort(text) {
    const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    const tiers = [];
    let i = 0;
    // Skip header lines
    while (i < lines.length && !lines[i].match(/^"?(IntervalTier|TextTier)"?$/)) i++;
    while (i < lines.length) {
        const tierClass = lines[i].replace(/"/g, '');
        const tierName = lines[i + 1] ? lines[i + 1].replace(/"/g, '') : '';
        i += 2;
        // skip xmin, xmax of tier
        i += 2;
        const count = parseInt(lines[i]); i++;
        const tier = { name: tierName, type: tierClass, items: [] };
        if (tierClass === 'IntervalTier') {
            for (let j = 0; j < count; j++) {
                const xmin = parseFloat(lines[i]); i++;
                const xmax = parseFloat(lines[i]); i++;
                const label = lines[i] ? lines[i].replace(/"/g, '') : ''; i++;
                tier.items.push({ xmin, xmax, text: label });
            }
        } else {
            for (let j = 0; j < count; j++) {
                const time = parseFloat(lines[i]); i++;
                const mark = lines[i] ? lines[i].replace(/"/g, '') : ''; i++;
                tier.items.push({ time, text: mark });
            }
        }
        tiers.push(tier);
    }
    return tiers;
}

// ── Application State ────────────────────────────────────────────
const state = {
    audioBuffer: null,
    audioFile: null,
    textGridData: null,
    viewStart: 0,
    viewEnd: 0,
    duration: 0,
    isPlaying: false,
    playbackTime: 0,
    audioContext: null,
    sourceNode: null,
    startedAt: 0,
    pausedAt: 0,
    // Selection range (click-drag)
    selection: null, // { start, end } in seconds
    isSelecting: false,
    selectionAnchor: 0,
};

// ── File Loading ─────────────────────────────────────────────────
const audioInput = document.getElementById('audio-input');
const textgridInput = document.getElementById('textgrid-input');

async function loadAudioFile(file) {
    if (!state.audioContext) {
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (state.isPlaying) stopPlayback();
    const arrayBuffer = await file.arrayBuffer();
    state.audioBuffer = await state.audioContext.decodeAudioData(arrayBuffer);
    state.duration = state.audioBuffer.duration;
    state.viewStart = 0;
    state.viewEnd = state.duration;
    state.pausedAt = 0;
    state.playbackTime = 0;
    state.selection = null;
    document.getElementById('load-audio-btn').textContent = file.name;
    resize();
}

function loadTextGridFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
        state.textGridData = parseTextGrid(reader.result);
        state.selection = null;
        document.getElementById('load-textgrid-btn').textContent = file.name;
        resize();
    };
    reader.readAsText(file);
}

document.getElementById('load-audio-btn').addEventListener('click', () => { audioInput.value = ''; audioInput.click(); });
document.getElementById('load-textgrid-btn').addEventListener('click', () => { textgridInput.value = ''; textgridInput.click(); });
audioInput.addEventListener('change', () => { if (audioInput.files.length) loadAudioFile(audioInput.files[0]); });
textgridInput.addEventListener('change', () => { if (textgridInput.files.length) loadTextGridFile(textgridInput.files[0]); });

// Drop files anywhere on the page
document.body.addEventListener('dragover', e => e.preventDefault());
document.body.addEventListener('drop', e => {
    e.preventDefault();
    for (const file of e.dataTransfer.files) {
        if (file.name.match(/\.(textgrid)$/i)) {
            loadTextGridFile(file);
        } else if (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|ogg|flac|m4a|aac)$/i)) {
            loadAudioFile(file);
        }
    }
});

// ── Viewer ───────────────────────────────────────────────────────
const RULER_HEIGHT = 28;
const WAVEFORM_HEIGHT = 200;
const TIER_HEIGHT = 100;
const TIER_GAP = 2;

let canvas, ctx, dpr, canvasW, canvasH;

function initViewer() {
    canvas = document.getElementById('main-canvas');
    ctx = canvas.getContext('2d');
    resize();
    window.addEventListener('resize', resize);
    setupControls();
    setupCanvasInteraction();
    render();
}

// Start viewer immediately on load
initViewer();

function resize() {
    dpr = window.devicePixelRatio || 1;
    const container = document.getElementById('canvas-container');
    canvasW = container.clientWidth;
    const tierCount = state.textGridData ? state.textGridData.length : 0;
    canvasH = RULER_HEIGHT + WAVEFORM_HEIGHT + tierCount * (TIER_HEIGHT + TIER_GAP) + 16;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    canvas.style.height = canvasH + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    render();
}

// ── Controls ─────────────────────────────────────────────────────
function setupControls() {
    document.getElementById('play-btn').addEventListener('click', togglePlay);
    document.getElementById('zoom-in-btn').addEventListener('click', () => zoom(0.8));
    document.getElementById('zoom-out-btn').addEventListener('click', () => zoom(1.25));
    document.getElementById('fit-btn').addEventListener('click', () => {
        state.viewStart = 0;
        state.viewEnd = state.duration;
        render();
    });
    document.getElementById('focus-btn').addEventListener('click', focusSelection);
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' || e.code === 'Tab') { e.preventDefault(); togglePlay(); }
        if (e.key === '+') { e.preventDefault(); zoom(0.8); }
        if (e.key === '=') { e.preventDefault(); focusSelection(); }
        if (e.key === '-') { e.preventDefault(); zoom(1.25); }
    });
}

function focusSelection() {
    if (state.selection) {
        const pad = (state.selection.end - state.selection.start) * 0.1;
        state.viewStart = Math.max(0, state.selection.start - pad);
        state.viewEnd = Math.min(state.duration, state.selection.end + pad);
        render();
    }
}

function togglePlay() {
    if (state.isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (!state.audioBuffer) return;
    state.audioContext.resume();
    state.sourceNode = state.audioContext.createBufferSource();
    state.sourceNode.buffer = state.audioBuffer;
    state.sourceNode.connect(state.audioContext.destination);
    // Play selection if exists, otherwise play what's in view
    const playStart = state.selection ? state.selection.start : state.viewStart;
    const playEnd = state.selection ? state.selection.end : state.viewEnd;
    const offset = (state.pausedAt >= playStart && state.pausedAt < playEnd)
        ? state.pausedAt : playStart;
    const duration = playEnd - offset;
    state.sourceNode.start(0, offset, duration);
    state.startedAt = state.audioContext.currentTime - offset;
    state.playbackEnd = playEnd;
    state.isPlaying = true;
    state.sourceNode.onended = () => {
        if (state.isPlaying) {
            state.isPlaying = false;
            state.pausedAt = state.selection ? state.selection.start : state.viewStart;
            updatePlayButton();
            render();
        }
    };
    updatePlayButton();
    animatePlayback();
}

function stopPlayback() {
    if (state.sourceNode) {
        state.sourceNode.onended = null;
        state.sourceNode.stop();
    }
    state.pausedAt = state.audioContext.currentTime - state.startedAt;
    state.isPlaying = false;
    updatePlayButton();
}

function updatePlayButton() {
    document.getElementById('play-btn').innerHTML = state.isPlaying ? '&#9646;&#9646; Pause' : '&#9654; Play';
}

function animatePlayback() {
    if (!state.isPlaying) return;
    state.playbackTime = state.audioContext.currentTime - state.startedAt;
    const endTime = state.playbackEnd || state.viewEnd;
    if (state.playbackTime >= endTime) {
        stopPlayback();
        state.pausedAt = state.selection ? state.selection.start : state.viewStart;
        render();
        return;
    }
    // Auto-scroll if cursor goes past view
    if (state.playbackTime > state.viewEnd) {
        const span = state.viewEnd - state.viewStart;
        state.viewStart = state.playbackTime;
        state.viewEnd = Math.min(state.viewStart + span, state.duration);
    }
    render();
    requestAnimationFrame(animatePlayback);
}

function zoom(factor) {
    const center = (state.viewStart + state.viewEnd) / 2;
    const span = (state.viewEnd - state.viewStart) * factor;
    const half = span / 2;
    state.viewStart = Math.max(0, center - half);
    state.viewEnd = Math.min(state.duration, center + half);
    render();
}

// ── Hit-test: find which tier entry was clicked ──────────────────
function hitTestTier(mouseX, mouseY) {
    if (!state.textGridData) return null;
    let yOffset = RULER_HEIGHT + WAVEFORM_HEIGHT;
    const clickTime = state.viewStart + (mouseX / canvasW) * (state.viewEnd - state.viewStart);

    for (const tier of state.textGridData) {
        yOffset += TIER_GAP;
        if (mouseY >= yOffset && mouseY < yOffset + TIER_HEIGHT) {
            if (tier.type === 'IntervalTier') {
                for (const iv of tier.items) {
                    if (clickTime >= iv.xmin && clickTime < iv.xmax) {
                        return { start: iv.xmin, end: iv.xmax };
                    }
                }
            } else {
                // Point tier — find nearest point, snap to it
                let nearest = null, bestDist = Infinity;
                for (const pt of tier.items) {
                    const d = Math.abs(pt.time - clickTime);
                    if (d < bestDist) { bestDist = d; nearest = pt; }
                }
                if (nearest) {
                    const pxDist = Math.abs(timeToX(nearest.time) - mouseX);
                    if (pxDist < 20) {
                        // Select a small region around the point for playback
                        return { start: nearest.time - 0.01, end: nearest.time + 0.01 };
                    }
                }
            }
            return null;
        }
        yOffset += TIER_HEIGHT;
    }
    return null;
}

// ── Canvas Interaction (scroll, drag, click, selection) ──────────
function setupCanvasInteraction() {
    let panning = false, dragStartX = 0, dragViewStart = 0, dragViewEnd = 0;

    canvas.addEventListener('mousedown', e => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (y < RULER_HEIGHT + WAVEFORM_HEIGHT) {
            // Start selection on waveform/ruler area
            const t = state.viewStart + (x / canvasW) * (state.viewEnd - state.viewStart);
            state.isSelecting = true;
            state.selectionAnchor = t;
            state.selection = null;
            seekTo(t);
            render();
            return;
        }
        // Click on tier area → select the clicked entry
        const clickedEntry = hitTestTier(x, y);
        if (clickedEntry) {
            if (e.shiftKey && state.selection) {
                const fullyInside = clickedEntry.start >= state.selection.start && clickedEntry.end <= state.selection.end;
                if (fullyInside) {
                    // Unselect: shrink from the nearest edge
                    const distToStart = Math.abs(clickedEntry.start - state.selection.start);
                    const distToEnd = Math.abs(state.selection.end - clickedEntry.end);
                    if (distToStart <= distToEnd) {
                        state.selection.start = clickedEntry.end;
                    } else {
                        state.selection.end = clickedEntry.start;
                    }
                    if (state.selection.end - state.selection.start < 0.005) {
                        state.selection = null;
                    }
                } else {
                    state.selection = {
                        start: Math.min(state.selection.start, clickedEntry.start),
                        end: Math.max(state.selection.end, clickedEntry.end)
                    };
                }
                if (state.selection) {
                    state.pausedAt = state.selection.start;
                    state.playbackTime = state.selection.start;
                }
            } else {
                state.selection = { start: clickedEntry.start, end: clickedEntry.end };
                state.pausedAt = clickedEntry.start;
                state.playbackTime = clickedEntry.start;
            }
            if (state.isPlaying) {
                stopPlayback();
                startPlayback();
            } else {
                render();
            }
            return;
        }
        // Pan on tier area (if no entry hit)
        panning = true;
        dragStartX = e.clientX;
        dragViewStart = state.viewStart;
        dragViewEnd = state.viewEnd;
    });

    window.addEventListener('mousemove', e => {
        if (state.isSelecting) {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const t = state.viewStart + (x / canvasW) * (state.viewEnd - state.viewStart);
            const start = Math.min(state.selectionAnchor, t);
            const end = Math.max(state.selectionAnchor, t);
            state.selection = { start: Math.max(0, start), end: Math.min(state.duration, end) };
            render();
            return;
        }
        if (!panning) return;
        const dx = e.clientX - dragStartX;
        const dt = -(dx / canvasW) * (dragViewEnd - dragViewStart);
        const span = dragViewEnd - dragViewStart;
        let newStart = dragViewStart + dt;
        let newEnd = dragViewEnd + dt;
        if (newStart < 0) { newStart = 0; newEnd = span; }
        if (newEnd > state.duration) { newEnd = state.duration; newStart = newEnd - span; }
        state.viewStart = newStart;
        state.viewEnd = newEnd;
        render();
    });

    window.addEventListener('mouseup', () => {
        if (state.isSelecting) {
            state.isSelecting = false;
            // If selection is too tiny (just a click), clear it — keep just the seek
            if (state.selection && (state.selection.end - state.selection.start) < 0.005) {
                state.selection = null;
                render();
            }
        }
        panning = false;
    });

    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) {
            // Zoom centered on mouse
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const ratio = x / canvasW;
            const span = state.viewEnd - state.viewStart;
            const factor = e.deltaY > 0 ? 1.08 : 1 / 1.08;
            const newSpan = Math.min(state.duration, Math.max(0.01, span * factor));
            const anchor = state.viewStart + ratio * span;
            state.viewStart = Math.max(0, anchor - ratio * newSpan);
            state.viewEnd = Math.min(state.duration, state.viewStart + newSpan);
            render();
        } else {
            // Scroll horizontally
            const span = state.viewEnd - state.viewStart;
            const dt = (e.deltaX || e.deltaY) * span * 0.002;
            state.viewStart = Math.max(0, Math.min(state.duration - span, state.viewStart + dt));
            state.viewEnd = state.viewStart + span;
            render();
        }
    }, { passive: false });
}

function seekTo(t) {
    t = Math.max(0, Math.min(state.duration, t));
    if (state.isPlaying) {
        stopPlayback();
        state.pausedAt = t;
        startPlayback();
    } else {
        state.pausedAt = t;
        state.playbackTime = t;
        render();
    }
}

// ── Rendering ────────────────────────────────────────────────────
function render() {
    ctx.clearRect(0, 0, canvasW, canvasH);
    drawRuler();
    drawWaveform();
    drawTiers();
    drawSelection();
    drawCursor();
    updateTimeDisplay();
}

function drawRuler() {
    const y = 0;
    ctx.fillStyle = '#16213e';
    ctx.fillRect(0, y, canvasW, RULER_HEIGHT);
    ctx.strokeStyle = '#3a3a5a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y + RULER_HEIGHT - 0.5);
    ctx.lineTo(canvasW, y + RULER_HEIGHT - 0.5);
    ctx.stroke();

    const span = state.viewEnd - state.viewStart;
    const step = niceStep(span, canvasW / 80);

    ctx.fillStyle = '#888';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';

    const first = Math.ceil(state.viewStart / step) * step;
    for (let t = first; t <= state.viewEnd; t += step) {
        const x = timeToX(t);
        ctx.beginPath();
        ctx.moveTo(x, y + RULER_HEIGHT - 10);
        ctx.lineTo(x, y + RULER_HEIGHT - 1);
        ctx.strokeStyle = '#555';
        ctx.stroke();
        ctx.fillText(formatTime(t), x, y + RULER_HEIGHT - 13);
    }
}

function contentBottom() {
    const tierCount = state.textGridData ? state.textGridData.length : 0;
    return RULER_HEIGHT + WAVEFORM_HEIGHT + tierCount * (TIER_GAP + TIER_HEIGHT);
}

function drawSelection() {
    if (!state.selection) return;
    const x1 = Math.max(0, timeToX(state.selection.start));
    const x2 = Math.min(canvasW, timeToX(state.selection.end));
    if (x2 <= x1) return;
    const yEnd = contentBottom();
    ctx.fillStyle = 'rgba(123, 123, 255, 0.15)';
    ctx.fillRect(x1, 0, x2 - x1, yEnd);
    // Selection boundary lines
    ctx.strokeStyle = 'rgba(123, 123, 255, 0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1 + 0.5, 0); ctx.lineTo(x1 + 0.5, yEnd);
    ctx.moveTo(x2 - 0.5, 0); ctx.lineTo(x2 - 0.5, yEnd);
    ctx.stroke();
}

function drawWaveform() {
    const y0 = RULER_HEIGHT;
    const h = WAVEFORM_HEIGHT;
    ctx.fillStyle = '#0f0f23';
    ctx.fillRect(0, y0, canvasW, h);

    if (!state.audioBuffer) return;
    const data = state.audioBuffer.getChannelData(0);
    const sr = state.audioBuffer.sampleRate;
    const startSample = Math.floor(state.viewStart * sr);
    const endSample = Math.ceil(state.viewEnd * sr);
    const samplesPerPixel = (endSample - startSample) / canvasW;

    const midY = y0 + h / 2;
    ctx.beginPath();
    ctx.strokeStyle = '#7b7bff';
    ctx.lineWidth = 1;

    if (samplesPerPixel < 1) {
        // Draw individual samples connected by lines
        for (let s = startSample; s < endSample && s < data.length; s++) {
            const x = timeToX(s / sr);
            const val = data[s];
            const py = midY - val * (h / 2 - 4);
            if (s === startSample) ctx.moveTo(x, py);
            else ctx.lineTo(x, py);
        }
        ctx.stroke();
    } else {
        // Draw min/max envelope
        for (let px = 0; px < canvasW; px++) {
            const s0 = Math.floor(startSample + px * samplesPerPixel);
            const s1 = Math.floor(s0 + samplesPerPixel);
            let min = 1, max = -1;
            for (let s = s0; s < s1 && s < data.length; s++) {
                if (data[s] < min) min = data[s];
                if (data[s] > max) max = data[s];
            }
            const y1 = midY - max * (h / 2 - 4);
            const y2 = midY - min * (h / 2 - 4);
            ctx.moveTo(px, y1);
            ctx.lineTo(px, y2);
        }
        ctx.stroke();
    }
}

function drawTiers() {
    if (!state.textGridData) return;
    let yOffset = RULER_HEIGHT + WAVEFORM_HEIGHT;
    const cursorTime = state.isPlaying ? state.playbackTime : state.pausedAt;

    for (const tier of state.textGridData) {
        yOffset += TIER_GAP;
        ctx.fillStyle = '#12122a';
        ctx.fillRect(0, yOffset, canvasW, TIER_HEIGHT);

        // Tier label
        ctx.fillStyle = '#666';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(tier.name, 6, yOffset + 13);

        if (tier.type === 'IntervalTier') {
            for (let i = 0; i < tier.items.length; i++) {
                const iv = tier.items[i];
                if (iv.xmax < state.viewStart || iv.xmin > state.viewEnd) continue;
                const x1 = Math.max(0, timeToX(iv.xmin));
                const x2 = Math.min(canvasW, timeToX(iv.xmax));
                const w = x2 - x1;
                const isActive = cursorTime >= iv.xmin && cursorTime < iv.xmax;

                // Highlight active interval
                if (isActive && iv.text) {
                    ctx.fillStyle = 'rgba(255, 68, 68, 0.12)';
                    ctx.fillRect(x1, yOffset, w, TIER_HEIGHT);
                }

                // Boundary lines
                ctx.strokeStyle = '#3a3a5a';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x1 + 0.5, yOffset);
                ctx.lineTo(x1 + 0.5, yOffset + TIER_HEIGHT);
                // Right border: only when no following item picks it up via its left border
                const next = tier.items[i + 1];
                if (!next || next.xmin > iv.xmax) {
                    ctx.moveTo(x2 - 0.5, yOffset);
                    ctx.lineTo(x2 - 0.5, yOffset + TIER_HEIGHT);
                }
                ctx.stroke();

                // Label
                if (iv.text && w > 8) {
                    ctx.fillStyle = isActive ? '#ff8888' : '#ccc';
                    ctx.font = isActive ? 'bold 12px sans-serif' : '12px sans-serif';
                    ctx.textAlign = 'center';
                    const tx = x1 + w / 2;
                    ctx.fillText(iv.text, tx, yOffset + TIER_HEIGHT / 2 + 4, w - 4);
                }
            }
        } else {
            // TextTier / point tier — highlight nearest point within a threshold
            let nearestPt = null, nearestDist = Infinity;
            for (const pt of tier.items) {
                const d = Math.abs(pt.time - cursorTime);
                if (d < nearestDist) { nearestDist = d; nearestPt = pt; }
            }
            const threshold = (state.viewEnd - state.viewStart) * 0.02;

            for (const pt of tier.items) {
                if (pt.time < state.viewStart || pt.time > state.viewEnd) continue;
                const x = timeToX(pt.time);
                const isActive = pt === nearestPt && nearestDist < threshold;

                ctx.strokeStyle = isActive ? '#ff4444' : '#ff7b7b';
                ctx.lineWidth = isActive ? 2.5 : 1.5;
                ctx.beginPath();
                ctx.moveTo(x, yOffset);
                ctx.lineTo(x, yOffset + TIER_HEIGHT);
                ctx.stroke();

                // Diamond marker
                ctx.fillStyle = isActive ? '#ff4444' : '#ff7b7b';
                const size = isActive ? 7 : 5;
                ctx.beginPath();
                ctx.moveTo(x, yOffset + TIER_HEIGHT / 2 - size);
                ctx.lineTo(x + size, yOffset + TIER_HEIGHT / 2);
                ctx.lineTo(x, yOffset + TIER_HEIGHT / 2 + size);
                ctx.lineTo(x - size, yOffset + TIER_HEIGHT / 2);
                ctx.closePath();
                ctx.fill();

                if (pt.text) {
                    ctx.fillStyle = isActive ? '#ff4444' : '#ff7b7b';
                    ctx.font = isActive ? 'bold 12px sans-serif' : '11px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(pt.text, x, yOffset + 13);
                }
            }
        }

        // Tier bottom border
        ctx.strokeStyle = '#2a2a4a';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, yOffset + TIER_HEIGHT - 0.5);
        ctx.lineTo(canvasW, yOffset + TIER_HEIGHT - 0.5);
        ctx.stroke();

        yOffset += TIER_HEIGHT;
    }
}

function drawCursor() {
    const t = state.isPlaying ? state.playbackTime : state.pausedAt;
    if (t < state.viewStart || t > state.viewEnd) return;
    const x = timeToX(t);
    ctx.strokeStyle = '#ff4444';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, contentBottom());
    ctx.stroke();
}

function updateTimeDisplay() {
    const t = state.isPlaying ? state.playbackTime : state.pausedAt;
    document.getElementById('time-display').textContent =
        `${formatTimeFull(t)} / ${formatTimeFull(state.duration)}`;
}

// ── Utilities ────────────────────────────────────────────────────
function timeToX(t) {
    return ((t - state.viewStart) / (state.viewEnd - state.viewStart)) * canvasW;
}

function niceStep(range, maxTicks) {
    const rough = range / maxTicks;
    const mag = Math.pow(10, Math.floor(Math.log10(rough)));
    const residual = rough / mag;
    let nice;
    if (residual <= 1.5) nice = 1;
    else if (residual <= 3.5) nice = 2;
    else if (residual <= 7.5) nice = 5;
    else nice = 10;
    return nice * mag;
}

function formatTime(t) {
    if (t >= 60) {
        const m = Math.floor(t / 60);
        const s = (t % 60).toFixed(1);
        return `${m}:${s.padStart(4, '0')}`;
    }
    return t.toFixed(2) + 's';
}

function formatTimeFull(t) {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    const ms = Math.floor((t % 1) * 1000);
    return `${m}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

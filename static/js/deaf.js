/**
 * HAND SLINGER - Deaf Person Mode (v2 - Improved)
 * ==================================================
 * Fixes:
 * - Higher confidence threshold to reject weak predictions
 * - Sliding window majority vote for stable predictions
 * - Better visual feedback and UI
 * - Easier to use with clear instructions
 * 
 * Uses:
 * - MediaPipe Hands (browser-side landmark detection)
 * - Server-side classification (landmarks → text)
 * - Translation history panel
 * - Fullscreen support (F key / ESC)
 */

// ============== DOM ELEMENTS ==============
const appLayout = document.getElementById('app-layout');
const cameraFeed = document.getElementById('camera-feed');
const landmarkCanvas = document.getElementById('landmark-canvas');
const cameraContainer = document.getElementById('camera-container');
const noCameraMsg = document.getElementById('no-camera-msg');
const detectionStatus = document.getElementById('detection-status');
const detectionText = document.getElementById('detection-text');
const predictionDisplay = document.getElementById('prediction-display');
const predictionWord = document.getElementById('prediction-word');
const predictionConfidence = document.getElementById('prediction-confidence');
const confidenceBar = document.getElementById('confidence-bar');
const translationHistory = document.getElementById('translation-history');
const translationEmpty = document.getElementById('translation-empty');
const clearHistoryBtn = document.getElementById('clear-history-btn');
const detectBtn = document.getElementById('detect-btn');
const detectIcon = document.getElementById('detect-icon');
const camBtn = document.getElementById('cam-btn');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const endBtn = document.getElementById('end-btn');
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const toast = document.getElementById('toast');

// ============== CONFIGURATION ==============
const PREDICTION_INTERVAL = 500;      // ms between API calls
const MIN_CONFIDENCE = 0.45;          // minimum confidence to show a prediction
const SLIDING_WINDOW_SIZE = 6;        // sliding window for majority vote
const STABLE_VOTES_NEEDED = 3;        // same sign N times in window to be "stable"
const HISTORY_DEBOUNCE = 2000;        // ms before same word can be added again

// ============== STATE ==============
let handsDetector = null;
let camera = null;
let isDetecting = true;
let isCameraOn = false;
let lastPredictionTime = 0;
let translationItems = [];

// Sliding window for prediction smoothing
let predictionWindow = [];   // recent predictions: [{label, confidence}, ...]
let lastHistoryWord = '';
let lastHistoryTime = 0;

const canvasCtx = landmarkCanvas.getContext('2d');

// ============== TOAST ==============
function showToast(message, duration = 3000) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}

// ============== LANDMARK NORMALIZATION ==============
/**
 * Normalize landmarks IDENTICALLY to Python training script:
 * 1. Translate so wrist (index 0) is at origin
 * 2. Scale so max distance from wrist = 1
 * Returns flat array padded to 126 features
 */
function normalizeLandmarks(multiHandLandmarks) {
    const allLandmarks = [];

    for (const handLandmarks of multiHandLandmarks) {
        const coords = handLandmarks.map(lm => [lm.x, lm.y, lm.z]);
        const wrist = [coords[0][0], coords[0][1], coords[0][2]];

        // Translate to wrist origin
        const translated = coords.map(c => [
            c[0] - wrist[0],
            c[1] - wrist[1],
            c[2] - wrist[2]
        ]);

        // Scale normalization
        let maxDist = 0;
        for (const c of translated) {
            const dist = Math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2);
            if (dist > maxDist) maxDist = dist;
        }

        const scaled = translated.map(c =>
            maxDist > 0
                ? [c[0] / maxDist, c[1] / maxDist, c[2] / maxDist]
                : c
        );

        for (const c of scaled) {
            allLandmarks.push(c[0], c[1], c[2]);
        }
    }

    // Pad to 126 (2 hands × 21 landmarks × 3 coords)
    while (allLandmarks.length < 126) {
        allLandmarks.push(0.0);
    }

    return allLandmarks.slice(0, 126);
}

// ============== MAJORITY VOTE (SLIDING WINDOW) ==============
/**
 * Adds a prediction to the sliding window and returns the
 * majority-vote result. This smooths out flickering predictions.
 */
function getMajorityPrediction(label, confidence) {
    predictionWindow.push({ label, confidence });

    // Keep only the last N predictions
    if (predictionWindow.length > SLIDING_WINDOW_SIZE) {
        predictionWindow.shift();
    }

    // Count votes per label
    const votes = {};
    let totalConf = {};
    for (const p of predictionWindow) {
        if (p.confidence >= MIN_CONFIDENCE) {
            votes[p.label] = (votes[p.label] || 0) + 1;
            totalConf[p.label] = (totalConf[p.label] || 0) + p.confidence;
        }
    }

    // Find the label with most votes
    let bestLabel = null;
    let bestVotes = 0;
    let bestAvgConf = 0;

    for (const [lbl, count] of Object.entries(votes)) {
        if (count > bestVotes || (count === bestVotes && totalConf[lbl] > bestAvgConf * count)) {
            bestLabel = lbl;
            bestVotes = count;
            bestAvgConf = totalConf[lbl] / count;
        }
    }

    if (bestLabel && bestVotes >= STABLE_VOTES_NEEDED) {
        return { label: bestLabel, confidence: bestAvgConf, stable: true };
    } else if (bestLabel) {
        return { label: bestLabel, confidence: bestAvgConf, stable: false };
    }

    return null;
}

// ============== MEDIAPIPE HANDS SETUP ==============
function initMediaPipeHands() {
    try {
        handsDetector = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
            }
        });

        handsDetector.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        handsDetector.onResults(onHandResults);

        statusText.textContent = 'Model loaded';
        showToast('✅ Hand detection ready! Show your signs to the camera.');

    } catch (err) {
        console.error('Failed to initialize MediaPipe Hands:', err);
        showToast('❌ Failed to load model. Please refresh the page.');
    }
}

// ============== CAMERA ==============
async function startCamera() {
    try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: 'user' },
            audio: false
        });

        cameraFeed.srcObject = mediaStream;

        await new Promise(resolve => {
            cameraFeed.onloadedmetadata = () => {
                landmarkCanvas.width = cameraFeed.videoWidth;
                landmarkCanvas.height = cameraFeed.videoHeight;
                resolve();
            };
        });

        noCameraMsg.style.display = 'none';
        isCameraOn = true;

        camera = new Camera(cameraFeed, {
            onFrame: async () => {
                if (handsDetector && isDetecting) {
                    await handsDetector.send({ image: cameraFeed });
                }
            },
            width: 1280,
            height: 720
        });
        camera.start();

        statusText.textContent = 'Active';
        detectionStatus.classList.add('active');
        detectionText.textContent = 'Detecting hands...';

    } catch (err) {
        console.error('Camera access denied:', err);
        noCameraMsg.style.display = 'flex';
        showToast('❌ Camera access denied. Please allow camera permissions.');
    }
}

// ============== HAND DETECTION RESULTS ==============
function onHandResults(results) {
    canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // Draw landmarks with vibrant colors
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: 'rgba(99, 102, 241, 0.7)',
                lineWidth: 3
            });
            drawLandmarks(canvasCtx, landmarks, {
                color: 'rgba(139, 92, 246, 0.9)',
                fillColor: 'rgba(236, 72, 153, 0.85)',
                lineWidth: 1,
                radius: 4
            });
        }

        // Update status
        cameraContainer.classList.add('detecting');
        detectionStatus.classList.add('active');
        const numHands = results.multiHandLandmarks.length;
        detectionText.textContent = `${numHands} hand${numHands > 1 ? 's' : ''} detected`;

        // Throttle prediction API calls
        const now = Date.now();
        if (now - lastPredictionTime > PREDICTION_INTERVAL) {
            lastPredictionTime = now;
            sendForPrediction(results.multiHandLandmarks);
        }

    } else {
        // No hands
        cameraContainer.classList.remove('detecting');
        detectionText.textContent = 'Show your hand sign...';

        // Clear sliding window when no hands detected for a while
        predictionWindow = [];
        predictionDisplay.style.display = 'none';
    }
}

// ============== PREDICTION ==============
async function sendForPrediction(multiHandLandmarks) {
    try {
        const landmarks = normalizeLandmarks(multiHandLandmarks);

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks: landmarks })
        });

        if (!response.ok) return;

        const data = await response.json();

        if (!data.prediction || data.confidence < MIN_CONFIDENCE) {
            // Low confidence - show analyzing state
            predictionDisplay.style.display = 'block';
            predictionWord.textContent = 'Analyzing...';
            predictionWord.style.fontSize = '20px';
            predictionConfidence.textContent = 'Hold your sign steady';
            confidenceBar.style.width = `${(data.confidence || 0) * 100}%`;
            confidenceBar.style.background = 'linear-gradient(90deg, #64748b, #94a3b8)';
            return;
        }

        // Use majority vote smoothing
        const majorityResult = getMajorityPrediction(data.prediction, data.confidence);

        if (majorityResult) {
            displayPrediction(majorityResult.label, majorityResult.confidence, majorityResult.stable, data.top_3);

            // Only add to history when prediction is stable
            if (majorityResult.stable) {
                const now = Date.now();
                // Debounce: don't add same word twice within HISTORY_DEBOUNCE ms
                if (majorityResult.label !== lastHistoryWord || (now - lastHistoryTime) > HISTORY_DEBOUNCE) {
                    addToHistory(majorityResult.label, majorityResult.confidence);
                    lastHistoryWord = majorityResult.label;
                    lastHistoryTime = now;
                }
            }
        }

    } catch (err) {
        console.error('Prediction failed:', err);
    }
}

function displayPrediction(word, confidence, isStable, top3 = []) {
    predictionDisplay.style.display = 'block';
    
    const displayWord = word.replace(/_/g, ' ');
    predictionWord.textContent = displayWord;
    predictionWord.style.fontSize = '28px';

    const pct = (confidence * 100).toFixed(0);
    predictionConfidence.textContent = isStable
        ? `✅ Confirmed: ${pct}% confidence`
        : `🔍 Detecting: ${pct}% (hold steady...)`;

    confidenceBar.style.width = `${pct}%`;

    if (confidence > 0.75) {
        confidenceBar.style.background = 'linear-gradient(90deg, #10b981, #22d3ee)';
    } else if (confidence > 0.55) {
        confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #10b981)';
    } else {
        confidenceBar.style.background = 'linear-gradient(90deg, #ef4444, #f59e0b)';
    }

    // Render top 3 candidates
    const chipsContainer = document.getElementById('top-candidates-container');
    if (chipsContainer && top3 && top3.length > 0) {
        chipsContainer.innerHTML = top3.map((cand, idx) => `
            <span class="candidate-chip ${idx === 0 ? 'rank-1' : ''}">
                ${cand.label.replace(/_/g, ' ')} (${(cand.confidence * 100).toFixed(0)}%)
            </span>
        `).join('');
    }
}

// ============== TRANSLATION HISTORY ==============
function addToHistory(word, confidence) {
    if (translationEmpty) {
        translationEmpty.style.display = 'none';
    }

    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const pct = (confidence * 100).toFixed(0);

    const item = document.createElement('div');
    item.className = 'translation-item';
    item.innerHTML = `
        <span class="translation-item-word">${word.replace(/_/g, ' ')}</span>
        <span class="translation-item-confidence">${pct}%</span>
        <span class="translation-item-time">${timeStr}</span>
    `;

    // Newest first
    const firstItem = translationHistory.querySelector('.translation-item');
    if (firstItem) {
        translationHistory.insertBefore(item, firstItem);
    } else {
        translationHistory.appendChild(item);
    }

    translationItems.push(item);
    updateHistoryBadge();
}

function updateHistoryBadge() {
    const badge = document.getElementById('history-count-badge');
    if (badge) {
        badge.textContent = translationItems.length;
    }
}

function clearHistory() {
    translationItems.forEach(item => item.remove());
    translationItems = [];
    predictionWindow = [];
    lastHistoryWord = '';
    lastHistoryTime = 0;
    if (translationEmpty) {
        translationEmpty.style.display = 'flex';
    }
    updateHistoryBadge();
    showToast('History cleared');
}

// ============== SIGNS GUIDE / DICTIONARY ==============
let allDictionarySigns = [];

function switchPanelTab(tab) {
    const historyBtn = document.getElementById('tab-history-btn');
    const dictBtn = document.getElementById('tab-dict-btn');
    const historyContent = document.getElementById('tab-history-content');
    const dictContent = document.getElementById('tab-dict-content');

    if (tab === 'history') {
        historyBtn.classList.add('active');
        dictBtn.classList.remove('active');
        historyContent.style.display = 'flex';
        dictContent.style.display = 'none';
    } else {
        dictBtn.classList.add('active');
        historyBtn.classList.remove('active');
        dictContent.style.display = 'flex';
        historyContent.style.display = 'none';
        if (allDictionarySigns.length === 0) {
            loadSignsDictionary();
        }
    }
}

async function loadSignsDictionary() {
    const list = document.getElementById('dict-signs-list');
    try {
        const res = await fetch('/api/labels');
        const data = await res.json();
        allDictionarySigns = data.labels || [];
        renderSignsDictionary(allDictionarySigns);
    } catch(err) {
        if (list) list.innerHTML = '<div style="color:var(--text-muted);padding:16px;">Failed to load signs list.</div>';
    }
}

function renderSignsDictionary(signs) {
    const list = document.getElementById('dict-signs-list');
    if (!list) return;
    if (!signs || signs.length === 0) {
        list.innerHTML = '<div style="color:var(--text-muted);padding:16px;text-align:center;">No matching signs found</div>';
        return;
    }
    list.innerHTML = signs.map(s => {
        const displayName = s.name.replace(/_/g, ' ');
        return `
            <div class="dict-item" onclick="openSignModal('${s.name}')">
                <img class="dict-item-thumb" src="/dataset/${encodeURIComponent(s.name)}/1.jpg" onerror="this.src='data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\' viewBox=\\'0 0 100 100\\'><text y=\\'.9em\\' font-size=\\'80\\'>🤟</text></svg>'" alt="${displayName}">
                <div style="flex:1;">
                    <div class="dict-item-name">${displayName}</div>
                    <div class="dict-item-sub">${s.image_count} frames · Click to view sample</div>
                </div>
            </div>
        `;
    }).join('');
}

function filterSignsDictionary() {
    const input = document.getElementById('dict-search-input');
    const q = input ? input.value.toLowerCase().trim() : '';
    if (!q) {
        renderSignsDictionary(allDictionarySigns);
    } else {
        const filtered = allDictionarySigns.filter(s =>
            s.name.toLowerCase().includes(q) || s.name.replace(/_/g, ' ').toLowerCase().includes(q)
        );
        renderSignsDictionary(filtered);
    }
}

async function openSignModal(signName) {
    try {
        const res = await fetch(`/api/get_sign/${encodeURIComponent(signName)}`);
        const data = await res.json();
        const displayName = signName.replace(/_/g, ' ');
        
        // Remove existing modal if any
        const existing = document.getElementById('sign-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.className = 'sign-modal-overlay';
        modal.id = 'sign-modal';
        modal.onclick = (e) => { if (e.target === modal) modal.remove(); };
        modal.innerHTML = `
            <div class="sign-modal-card">
                <img src="${data.image_url}" alt="${displayName}" style="max-height:260px;object-fit:contain;">
                <h3>${displayName}</h3>
                <p>Perform this hand gesture clearly towards the camera to translate into text.</p>
                <button onclick="document.getElementById('sign-modal').remove()">Close</button>
            </div>
        `;
        document.body.appendChild(modal);
    } catch(err) {
        showToast('Could not load sign preview');
    }
}

// ============== FULLSCREEN ==============
function enterFullscreen() {
    const elem = appLayout;
    if (elem.requestFullscreen) {
        elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) {
        elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) {
        elem.msRequestFullscreen();
    }
    showToast('Fullscreen mode — press ESC to exit');
}

function toggleFullscreen() {
    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else {
        enterFullscreen();
    }
}

// ============== EVENT LISTENERS ==============

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        enterFullscreen();
    }
});

// Detection toggle
detectBtn.addEventListener('click', () => {
    isDetecting = !isDetecting;
    if (isDetecting) {
        detectBtn.classList.add('active');
        detectIcon.textContent = '🧠';
        detectionStatus.classList.add('active');
        detectionText.textContent = 'Detecting hands...';
        showToast('✅ Detection enabled');
    } else {
        detectBtn.classList.remove('active');
        detectIcon.textContent = '🧠';
        detectionStatus.classList.remove('active');
        detectionText.textContent = 'Detection paused';
        predictionDisplay.style.display = 'none';
        cameraContainer.classList.remove('detecting');
        predictionWindow = [];
        showToast('⏸️ Detection paused');
    }
});

// Camera toggle
camBtn.addEventListener('click', () => {
    showToast('📷 Camera is required for sign detection');
});

// Fullscreen button
fullscreenBtn.addEventListener('click', toggleFullscreen);

// Clear history
clearHistoryBtn.addEventListener('click', clearHistory);

// End / Leave
endBtn.addEventListener('click', () => {
    if (camera) camera.stop();
    window.location.href = '/';
});

// ============== INITIALIZATION ==============
document.addEventListener('DOMContentLoaded', () => {
    statusText.textContent = 'Loading model...';
    initMediaPipeHands();
    setTimeout(() => startCamera(), 500);
});

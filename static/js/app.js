/**
 * HAND SLINGER — Unified Single-Website Application
 * ==================================================
 * Features:
 * 1. Mode Switcher: Hearing Mode | Deaf Mode | Two-Way Meet
 * 2. Hearing Mode: Speech & Text → Sign language images (first frame)
 * 3. Deaf Mode: MediaPipe hand tracking → 95.48% Ensemble AI prediction
 * 4. Two-Way Meet: Dual interactive Google Meet call between both users
 * 5. Fullscreen support: Press 'F' to enter, 'ESC' to exit
 * 6. Reference Dictionary: Search and inspect all 114 sign words
 */

// ============== GLOBAL STATE ==============
let currentMode = 'normal';
let allSignLabels = [];
let toastTimeout = null;

// MediaPipe state
let handsDetector = null;
let faceDetector = null;
let deafCamera = null;
let dualCamera = null;
let isDeafDetecting = true;
let isDeafCamOn = false;
let isNormalCamOn = false;
let lastPredictionTime = 0;

// Hand Sign Stability & Hold-to-Detect State (Prevents hasty/mistaken predictions)
let prevHandLandmarks = null;
let holdStartTime = 0;
let holdProgress = 0;
const REQUIRED_HOLD_MS = 1400; // User must hold hand sign steady for 1.4s before prediction
const MAX_JITTER_ALLOWED = 0.040; // Strict steadiness threshold
let isRequestingPrediction = false;

// Head Movement Confirmation State (Nod = Correct / Confirm, Shake = Wrong / Cancel)
let awaitingHeadConfirmation = false;
let pendingProposedWord = null;
let confirmationCooldownUntil = 0;
let headHistory = []; // [{ x, y, time }]
let offscreenCanvas = null;
let offscreenCtx = null;

// Confirmed Translated Sentence (Displayed below the screen)
let confirmedSentenceWords = [];

// Prediction smoothing (Dual mode sliding window)
const PREDICTION_INTERVAL = 220;
const MIN_CONFIDENCE = 0.12;
const ACCEPT_CONFIDENCE_THRESHOLD = 0.35;
const SLIDING_WINDOW_SIZE = 4;
const STABLE_VOTES_NEEDED = 2;
let predictionWindow = [];
let lastHistoryWord = '';
let lastHistoryTime = 0;

// Translation & History items
let deafHistoryItems = [];
let normalSignCards = [];
let dualDialogueItems = [];

// Speech recognition
let speechRecognition = null;
let isListening = false;
let isDualListening = false;

// ============== DOM INITIALIZATION ==============
document.addEventListener('DOMContentLoaded', () => {
    // Read query parameter if provided: ?mode=normal or ?mode=deaf or ?mode=dual
    const urlParams = new URLSearchParams(window.location.search);
    const modeParam = urlParams.get('mode');
    if (modeParam && ['normal', 'deaf', 'dual'].includes(modeParam)) {
        setAppMode(modeParam);
    } else {
        setAppMode('normal');
    }

    // Initialize speech recognition
    initSpeechEngine();

    // Initialize MediaPipe hands & face detection
    initMediaPipeHands();
    initMediaPipeFace();

    // Fetch dataset labels for dictionary & auto-complete
    fetchDatasetLabels();

    // Global keyboard shortcuts (F for fullscreen, Enter/Escape for confirmation)
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        if (awaitingHeadConfirmation) {
            if (e.key === 'Enter') {
                e.preventDefault();
                resolveHeadConfirmation(true);
                return;
            } else if (e.key === 'Escape' || e.key === 'Backspace' || e.key === ' ') {
                e.preventDefault();
                resolveHeadConfirmation(false);
                return;
            }
        }

        if (e.key === 'f' || e.key === 'F') {
            e.preventDefault();
            toggleAppFullscreen();
        }
    });
});

// ============== TOAST NOTIFICATIONS ==============
function showToast(msg, duration = 3000) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => toast.classList.remove('show'), duration);
}

// ============== MODE SWITCHER ==============
function setAppMode(mode) {
    currentMode = mode;

    // Update buttons
    ['normal', 'deaf', 'dual'].forEach(m => {
        const btn = document.getElementById(`btn-mode-${m}`);
        const sec = document.getElementById(`section-${m}`);
        if (btn) btn.classList.toggle('active', m === mode);
        if (sec) sec.style.display = (m === mode) ? 'block' : 'none';
    });

    const statusText = document.getElementById('status-text');
    if (statusText) {
        if (mode === 'normal') statusText.textContent = 'Hearing Mode';
        else if (mode === 'deaf') statusText.textContent = 'Deaf Mode';
        else statusText.textContent = 'Two-Way Meet';
    }

    // Stop streams not in use to save resources
    if (mode === 'normal') {
        stopDeafCamera();
        stopDualCamera();
    } else if (mode === 'deaf') {
        stopNormalCamera();
        stopDualCamera();
        startDeafCamera();
    } else if (mode === 'dual') {
        stopNormalCamera();
        stopDeafCamera();
        startDualCamera();
    }
}

// ============== FULLSCREEN TOGGLE ==============
function toggleAppFullscreen() {
    const layout = document.getElementById('app-layout') || document.documentElement;
    if (!document.fullscreenElement) {
        if (layout.requestFullscreen) layout.requestFullscreen();
        else if (layout.webkitRequestFullscreen) layout.webkitRequestFullscreen();
        showToast('⛶ Fullscreen active (Press ESC to exit)');
    } else {
        if (document.exitFullscreen) document.exitFullscreen();
        showToast('Exited fullscreen');
    }
}

// ============== MEDIAPIPE HANDS INITIALIZATION ==============
function initMediaPipeHands() {
    try {
        handsDetector = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
        });

        handsDetector.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.65,
            minTrackingConfidence: 0.65
        });

        handsDetector.onResults(onHandResults);
    } catch (err) {
        console.error('MediaPipe Hands setup error:', err);
    }
}

// ============== MEDIAPIPE FACE DETECTION INITIALIZATION ==============
function initMediaPipeFace() {
    try {
        if (typeof FaceDetection !== 'undefined') {
            faceDetector = new FaceDetection({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4.1646425229/${file}`
            });
            faceDetector.setOptions({
                model: 'short',
                minDetectionConfidence: 0.5
            });
            faceDetector.onResults(onFaceResults);
            console.log('[OK] MediaPipe Face Detection initialized');
        } else {
            console.log('[INFO] FaceDetection CDN loading or using optical tracker fallback');
        }
    } catch (err) {
        console.warn('MediaPipe Face Detection init error (fallback optical tracker active):', err);
    }
}

// Synthesizer Audio Ping (Positive chord on confirm, gentle drop on cancel)
function playChime(isPositive = true) {
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        const ctx = new AudioCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        if (isPositive) {
            // Confirm chime: E5 -> G#5 -> B5 upward chord
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(659.25, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(987.77, ctx.currentTime + 0.12);
            gain.gain.setValueAtTime(0.18, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.28);
            osc.start();
            osc.stop(ctx.currentTime + 0.28);
        } else {
            // Reject chime: D4 -> G3 downward tone
            osc.type = 'sine';
            osc.frequency.setValueAtTime(293.66, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(196.00, ctx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.14, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.22);
            osc.start();
            osc.stop(ctx.currentTime + 0.22);
        }
    } catch (e) {
        // AudioContext might require user interaction on some browsers
    }
}

// Coordinate normalization matching Python training
function normalizeLandmarks(multiHandLandmarks) {
    const allLandmarks = [];
    for (const handLandmarks of multiHandLandmarks) {
        const coords = handLandmarks.map(lm => [lm.x, lm.y, lm.z]);
        const wrist = [coords[0][0], coords[0][1], coords[0][2]];

        // Wrist-relative translation
        const translated = coords.map(c => [
            c[0] - wrist[0],
            c[1] - wrist[1],
            c[2] - wrist[2]
        ]);

        // Max distance scaling
        let maxDist = 0;
        for (const c of translated) {
            const dist = Math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2);
            if (dist > maxDist) maxDist = dist;
        }

        const scaled = translated.map(c =>
            maxDist > 0 ? [c[0] / maxDist, c[1] / maxDist, c[2] / maxDist] : c
        );

        for (const c of scaled) {
            allLandmarks.push(c[0], c[1], c[2]);
        }
    }

    // Pad to 126 coords
    while (allLandmarks.length < 126) allLandmarks.push(0.0);
    return allLandmarks.slice(0, 126);
}

// Majority vote smoothing
function getMajorityPrediction(label, confidence) {
    predictionWindow.push({ label, confidence });
    if (predictionWindow.length > SLIDING_WINDOW_SIZE) {
        predictionWindow.shift();
    }

    const counts = {};
    const sumConf = {};
    for (const p of predictionWindow) {
        if (p.confidence >= MIN_CONFIDENCE) {
            counts[p.label] = (counts[p.label] || 0) + 1;
            sumConf[p.label] = (sumConf[p.label] || 0) + p.confidence;
        }
    }

    let topLabel = null;
    let topCount = 0;
    for (const [lbl, cnt] of Object.entries(counts)) {
        if (cnt > topCount) {
            topCount = cnt;
            topLabel = lbl;
        }
    }

    if (topLabel) {
        const avgConf = sumConf[topLabel] / topCount;
        return { label: topLabel, confidence: avgConf, stable: topCount >= STABLE_VOTES_NEEDED };
    }
    return null;
}

// MediaPipe frame results callback
function onHandResults(results) {
    if (currentMode === 'deaf') {
        renderDeafSkeleton(results);
    } else if (currentMode === 'dual') {
        renderDualSkeleton(results);
    }
}

// Landmark displacement calculation for stability detection
function computeHandDisplacement(prevLms, currLms) {
    if (!prevLms || !currLms || prevLms.length === 0 || currLms.length === 0) return 0;
    const keyIndices = [0, 4, 8, 12, 16, 20];
    let totalDist = 0;
    let count = 0;
    for (const idx of keyIndices) {
        if (prevLms[idx] && currLms[idx]) {
            const dx = currLms[idx].x - prevLms[idx].x;
            const dy = currLms[idx].y - prevLms[idx].y;
            totalDist += Math.sqrt(dx * dx + dy * dy);
            count++;
        }
    }
    return count > 0 ? (totalDist / count) : 0;
}

// Validates that the detected landmarks represent a real, visible hand
function isValidHand(landmarks) {
    if (!landmarks || landmarks.length < 21) return false;
    const xs = landmarks.map(p => p.x);
    const ys = landmarks.map(p => p.y);
    const spanX = Math.max(...xs) - Math.min(...xs);
    const spanY = Math.max(...ys) - Math.min(...ys);
    // Must occupy a reasonable area on screen to prevent false background artifacts
    return (spanX >= 0.075 || spanY >= 0.075);
}

// MediaPipe frame results callback
function onHandResults(results) {
    if (currentMode === 'deaf') {
        renderDeafSkeleton(results);
    } else if (currentMode === 'dual') {
        renderDualSkeleton(results);
    }
}

function renderDeafSkeleton(results) {
    const canvas = document.getElementById('deaf-landmark-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const statusElem = document.getElementById('deaf-detection-text');
    const statusBox = document.getElementById('deaf-detection-status');
    const container = document.getElementById('deaf-camera-container');
    const scanLine = document.getElementById('deaf-scan-line');
    const scanStatusText = document.getElementById('deaf-scan-status-text');
    const scanBadge = document.getElementById('deaf-scan-badge');

    // If currently waiting for user to nod/shake, do not track hands
    if (awaitingHeadConfirmation) {
        if (statusElem) statusElem.textContent = 'Awaiting head response (Nod / Shake)...';
        if (scanLine) scanLine.style.width = '0%';
        if (scanBadge) scanBadge.textContent = 'Waiting';
        return;
    }

    // Cooldown right after confirming/cancelling to allow repositioning
    const now = Date.now();
    if (now < confirmationCooldownUntil) {
        if (statusElem) statusElem.textContent = 'Get ready for next sign...';
        if (scanLine) scanLine.style.width = '0%';
        if (scanStatusText) scanStatusText.textContent = 'Waiting for next hand sign...';
        if (scanBadge) scanBadge.textContent = 'Ready';
        prevHandLandmarks = null;
        holdStartTime = 0;
        holdProgress = 0;
        isRequestingPrediction = false;
        return;
    }

    const hasHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
    const primaryHand = hasHands ? results.multiHandLandmarks[0] : null;

    // RULE: If NO valid hand is detected, DO NOT LOAD / DO NOT CALL PREDICT AT ALL!
    if (!hasHands || !primaryHand || !isValidHand(primaryHand)) {
        if (statusBox) statusBox.classList.remove('active');
        if (container) container.classList.remove('detecting');
        if (statusElem) statusElem.textContent = 'Waiting for hand sign...';
        if (scanLine) scanLine.style.width = '0%';
        if (scanStatusText) scanStatusText.textContent = 'Waiting for hand sign...';
        if (scanBadge) scanBadge.textContent = 'No Hand';
        prevHandLandmarks = null;
        holdStartTime = 0;
        holdProgress = 0;
        isRequestingPrediction = false;
        return;
    }

    // Valid hand detected: Draw landmark skeleton
    for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
            color: 'rgba(99, 102, 241, 0.75)',
            lineWidth: 3
        });
        drawLandmarks(ctx, landmarks, {
            color: 'rgba(139, 92, 246, 0.95)',
            fillColor: 'rgba(34, 211, 238, 0.9)',
            lineWidth: 1,
            radius: 4
        });
    }

    if (statusBox) statusBox.classList.add('active');
    if (container) container.classList.add('detecting');
    const numHands = results.multiHandLandmarks.length;
    if (statusElem) statusElem.textContent = `${numHands} hand${numHands > 1 ? 's' : ''} detected`;

    // Measure hand steadiness (displacement between consecutive frames)
    let displacement = 0;
    if (prevHandLandmarks) {
        displacement = computeHandDisplacement(prevHandLandmarks, primaryHand);
    }
    prevHandLandmarks = primaryHand;

    if (displacement > MAX_JITTER_ALLOWED) {
        // Hand is moving or forming the gesture: DO NOT LOAD, reset progress
        holdStartTime = 0;
        holdProgress = 0;
        if (scanLine) scanLine.style.width = '0%';
        if (scanStatusText) scanStatusText.textContent = 'Aligning sign... hold still';
        if (scanBadge) scanBadge.textContent = 'Moving';
        isRequestingPrediction = false;
    } else {
        // Hand is held steady: Start charging the detection line!
        if (!holdStartTime) holdStartTime = now;
        const elapsed = now - holdStartTime;
        holdProgress = Math.min(1.0, elapsed / REQUIRED_HOLD_MS);

        const pct = Math.round(holdProgress * 100);
        if (scanLine) scanLine.style.width = `${pct}%`;
        if (scanBadge) scanBadge.textContent = `${pct}%`;

        if (holdProgress < 1.0) {
            if (statusElem) statusElem.textContent = `Holding sign steady... ${pct}%`;
            if (scanStatusText) scanStatusText.textContent = `Detecting sign... keep holding still`;
        } else if (!isRequestingPrediction) {
            // Reached 100% steady hold! Make exactly ONE predict request
            isRequestingPrediction = true;
            if (statusElem) statusElem.textContent = 'Recognizing sign...';
            if (scanStatusText) scanStatusText.textContent = 'Analyzing sign gesture...';
            if (scanBadge) scanBadge.textContent = '100%';
            sendLandmarksForPrediction(results.multiHandLandmarks, 'deaf');
        }
    }
}

function renderDualSkeleton(results) {
    const canvas = document.getElementById('dual-landmark-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                color: 'rgba(99, 102, 241, 0.7)',
                lineWidth: 2
            });
            drawLandmarks(ctx, landmarks, {
                color: 'rgba(236, 72, 153, 0.9)',
                fillColor: 'rgba(34, 211, 238, 0.9)',
                lineWidth: 1,
                radius: 3
            });
        }
        const now = Date.now();
        if (now - lastPredictionTime > PREDICTION_INTERVAL) {
            lastPredictionTime = now;
            sendLandmarksForPrediction(results.multiHandLandmarks, 'dual');
        }
    }
}

// Server prediction API call
async function sendLandmarksForPrediction(multiHandLandmarks, targetMode) {
    try {
        const landmarks = normalizeLandmarks(multiHandLandmarks);
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks })
        });
        isRequestingPrediction = false;
        if (!res.ok) return;
        const data = await res.json();

        if (targetMode === 'deaf') {
            handleDeafPrediction(data);
        } else if (targetMode === 'dual') {
            handleDualPrediction(data);
        }
    } catch (err) {
        isRequestingPrediction = false;
        console.error('Prediction request error:', err);
    }
}

// Deaf Mode: Proposes detected sign and waits for HEAD MOVEMENT confirmation
function handleDeafPrediction(data) {
    const proposed = (data.prediction || '').trim();
    if (!proposed) {
        holdStartTime = 0;
        holdProgress = 0;
        return;
    }

    // Enter Head Confirmation state
    awaitingHeadConfirmation = true;
    pendingProposedWord = proposed;
    headHistory = [];

    // Show Confirmation Overlay with proposed word
    const overlay = document.getElementById('deaf-confirm-overlay');
    const wordElem = document.getElementById('deaf-confirm-word');
    const gaugeStatus = document.getElementById('head-gauge-status');
    const gaugeDot = document.getElementById('head-gauge-dot');

    if (wordElem) wordElem.textContent = proposed.replace(/_/g, ' ');
    if (gaugeStatus) gaugeStatus.textContent = 'Move head UP & DOWN to Confirm (or Shake to Cancel)';
    if (gaugeDot) {
        gaugeDot.style.transform = 'translate(0px, 0px)';
        gaugeDot.style.background = '#38bdf8';
    }
    if (overlay) overlay.style.display = 'flex';
}

// ============ HEAD MOVEMENT GESTURE RECOGNITION ============
function onFaceResults(results) {
    if (!awaitingHeadConfirmation) return;
    if (results.detections && results.detections.length > 0) {
        const det = results.detections[0];
        // Nose tip is landmark 2 in MediaPipe FaceDetection
        let x, y;
        if (det.landmarks && det.landmarks[2]) {
            x = det.landmarks[2].x;
            y = det.landmarks[2].y;
        } else if (det.boundingBox) {
            x = det.boundingBox.xCenter;
            y = det.boundingBox.yCenter;
        }
        if (x !== undefined && y !== undefined) {
            trackHeadCoordinates(x, y);
            return;
        }
    }

    // Fallback: optical tracking on video element
    const video = document.getElementById('deaf-camera-feed');
    if (video) processOpticalHeadMovement(video);
}

// Fallback: Fast optical brightness centroid tracker in upper-center region
function processOpticalHeadMovement(video) {
    if (!video || video.readyState < 2) return;
    if (!offscreenCanvas) {
        offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 64;
        offscreenCanvas.height = 64;
        offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
    }

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    // Upper-center quadrant (head position)
    offscreenCtx.drawImage(
        video,
        vw * 0.25, vh * 0.05, vw * 0.5, vh * 0.45,
        0, 0, 64, 64
    );
    const imgData = offscreenCtx.getImageData(0, 0, 64, 64);
    const data = imgData.data;

    let totalWeight = 0;
    let sumX = 0;
    let sumY = 0;
    for (let y = 0; y < 64; y += 2) {
        for (let x = 0; x < 64; x += 2) {
            const idx = (y * 64 + x) * 4;
            const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            totalWeight += lum;
            sumX += x * lum;
            sumY += y * lum;
        }
    }

    if (totalWeight > 0) {
        const cx = sumX / (totalWeight * 64);
        const cy = sumY / (totalWeight * 64);
        trackHeadCoordinates(cx, cy);
    }
}

// Track head trajectory & classify Nod (Up & Down) vs Shake (Left & Right)
function trackHeadCoordinates(x, y) {
    const now = Date.now();
    headHistory.push({ x, y, time: now });

    // Keep history window to ~750ms
    while (headHistory.length > 22 || (headHistory.length > 0 && now - headHistory[0].time > 750)) {
        headHistory.shift();
    }

    if (headHistory.length < 5) return;

    const avgX = headHistory.reduce((s, p) => s + p.x, 0) / headHistory.length;
    const avgY = headHistory.reduce((s, p) => s + p.y, 0) / headHistory.length;

    // Relative displacement for live gauge dot
    const dx = (x - avgX) * 140;
    const dy = (y - avgY) * 140;
    const gaugeDot = document.getElementById('head-gauge-dot');
    const gaugeStatus = document.getElementById('head-gauge-status');

    if (gaugeDot) {
        const clampedX = Math.max(-55, Math.min(55, -dx));
        const clampedY = Math.max(-10, Math.min(10, dy));
        gaugeDot.style.transform = `translate(${clampedX}px, ${clampedY}px)`;
    }

    const ys = headHistory.map(p => p.y);
    const xs = headHistory.map(p => p.x);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const rangeY = maxY - minY;
    const rangeX = maxX - minX;

    // Count vertical reversals (peaks/valleys)
    let yReversals = 0;
    for (let i = 1; i < ys.length - 1; i++) {
        if ((ys[i] > ys[i-1] && ys[i] > ys[i+1]) || (ys[i] < ys[i-1] && ys[i] < ys[i+1])) {
            yReversals++;
        }
    }

    // Count horizontal reversals
    let xReversals = 0;
    for (let i = 1; i < xs.length - 1; i++) {
        if ((xs[i] > xs[i-1] && xs[i] > xs[i+1]) || (xs[i] < xs[i-1] && xs[i] < xs[i+1])) {
            xReversals++;
        }
    }

    // 1. NOD DETECTED (Up and Down motion) -> YES / CORRECT
    if (rangeY > 0.032 && yReversals >= 1 && rangeY > rangeX * 1.15) {
        if (gaugeStatus) gaugeStatus.innerHTML = '<strong style="color:#34d399">↕️ Nod Detected! Confirmed ✓</strong>';
        if (gaugeDot) gaugeDot.style.background = '#10b981';
        resolveHeadConfirmation(true);
        return;
    }

    // 2. SHAKE DETECTED (Left and Right motion) -> NO / WRONG
    if (rangeX > 0.040 && xReversals >= 1 && rangeX > rangeY * 1.15) {
        if (gaugeStatus) gaugeStatus.innerHTML = '<strong style="color:#f87171">↔️ Shake Detected! Cancelled ✗</strong>';
        if (gaugeDot) gaugeDot.style.background = '#ef4444';
        resolveHeadConfirmation(false);
        return;
    }

    if (gaugeStatus) {
        if (rangeY > rangeX) {
            gaugeStatus.textContent = 'Moving vertically (Nodding)...';
        } else if (rangeX > rangeY) {
            gaugeStatus.textContent = 'Moving horizontally (Shaking)...';
        } else {
            gaugeStatus.textContent = 'Nod head (Up & Down) to confirm';
        }
    }
}

// Resolves the head confirmation (Nod = true, Shake/Cancel = false)
function resolveHeadConfirmation(isConfirmed) {
    if (!awaitingHeadConfirmation) return;
    awaitingHeadConfirmation = false;
    headHistory = [];

    const overlay = document.getElementById('deaf-confirm-overlay');
    if (overlay) overlay.style.display = 'none';

    if (isConfirmed && pendingProposedWord) {
        playChime(true);
        addWordToSentence(pendingProposedWord);
        showToast(`✅ Confirmed: "${pendingProposedWord.replace(/_/g, ' ')}"`);
    } else {
        playChime(false);
        showToast('❌ Cancelled. Ready for next sign.');
    }

    pendingProposedWord = null;
    confirmationCooldownUntil = Date.now() + 1500;
    holdStartTime = 0;
    holdProgress = 0;
    prevHandLandmarks = null;
    isRequestingPrediction = false;
    const scanLine = document.getElementById('deaf-scan-line');
    if (scanLine) scanLine.style.width = '0%';
    const scanBadge = document.getElementById('deaf-scan-badge');
    if (scanBadge) scanBadge.textContent = 'Ready';
}

// ============ TRANSLATED SENTENCE (BELOW SCREEN) ============
function addWordToSentence(word) {
    const clean = word.replace(/_/g, ' ').toUpperCase();
    confirmedSentenceWords.push(clean);
    renderConfirmedSentence();
}

function renderConfirmedSentence() {
    const sentenceElem = document.getElementById('deaf-translated-sentence');
    const badge = document.getElementById('deaf-word-count-badge');
    const strip = document.getElementById('deaf-confirmed-words-strip');

    const count = confirmedSentenceWords.length;
    if (badge) badge.textContent = `${count} word${count === 1 ? '' : 's'}`;

    if (!sentenceElem) return;

    if (count === 0) {
        sentenceElem.innerHTML = '<span class="subtitles-empty">Perform a hand sign and nod your head (Up & Down) to confirm words here...</span>';
        if (strip) strip.innerHTML = '';
        return;
    }

    sentenceElem.innerHTML = confirmedSentenceWords.map((w, idx) => {
        const isRecent = idx === count - 1;
        return `<span class="confirmed-word-chip ${isRecent ? 'recent' : ''}">${w}</span>`;
    }).join(' ');

    sentenceElem.scrollLeft = sentenceElem.scrollWidth;
}

function clearCurrentSentence() {
    confirmedSentenceWords = [];
    renderConfirmedSentence();
    showToast('Translated text cleared');
}

function copyCurrentSentence() {
    if (confirmedSentenceWords.length === 0) {
        showToast('No translated text to copy');
        return;
    }
    const fullText = confirmedSentenceWords.join(' ');
    navigator.clipboard.writeText(fullText).then(() => {
        showToast(`📋 Copied: "${fullText}"`);
    }).catch(() => {
        showToast('Failed to copy text');
    });
}

function speakCurrentSentence() {
    if (confirmedSentenceWords.length === 0) {
        showToast('No translated text to speak');
        return;
    }
    const fullText = confirmedSentenceWords.join(' ').toLowerCase();
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(fullText);
        utter.rate = 0.95;
        window.speechSynthesis.speak(utter);
        showToast(`🔊 Speaking: "${fullText}"`);
    } else {
        showToast('Speech synthesis not supported in browser');
    }
}

function handleDualPrediction(data) {
    const topCandidate = (data.top_3 && data.top_3.length > 0)
        ? data.top_3[0]
        : { label: data.prediction || '', confidence: data.confidence || 0 };

    const highestWeightWord = topCandidate.label;
    const highestWeightConf = topCandidate.confidence;
    if (!highestWeightWord || highestWeightConf < MIN_CONFIDENCE) return;

    const displayWord = highestWeightWord.replace(/_/g, ' ');
    const pct = (highestWeightConf * 100).toFixed(0);

    const captionBar = document.getElementById('dual-live-caption');
    const predDisp = document.getElementById('dual-prediction-display');
    const predWord = document.getElementById('dual-pred-word');
    const predConf = document.getElementById('dual-pred-conf');

    if (predDisp) predDisp.style.display = 'block';
    if (predWord) predWord.textContent = displayWord;
    if (predConf) predConf.textContent = `${pct}%`;

    // RULE: Accept if confidence > 35%
    if (highestWeightConf > ACCEPT_CONFIDENCE_THRESHOLD) {
        if (captionBar) captionBar.innerHTML = `<strong>🤟 Deaf:</strong> "${displayWord}" (${pct}% - Accepted)`;

        const now = Date.now();
        if (highestWeightWord !== lastHistoryWord || (now - lastHistoryTime) > 1800) {
            addDualDialogue('deaf', displayWord, highestWeightConf);
            lastHistoryWord = highestWeightWord;
            lastHistoryTime = now;
        }
    } else {
        if (captionBar) captionBar.innerHTML = `<em>Detecting: "${displayWord}" (${pct}%)...</em>`;
    }
}

// ============== DEAF MODE CAMERA & CONTROLS ==============
async function startDeafCamera() {
    const video = document.getElementById('deaf-camera-feed');
    const canvas = document.getElementById('deaf-landmark-canvas');
    const noCam = document.getElementById('deaf-no-camera-msg');
    const camBtn = document.getElementById('deaf-cam-btn');

    if (!video || !canvas) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: 'user' },
            audio: false
        });
        video.srcObject = stream;
        await new Promise(r => { video.onloadedmetadata = r; });
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        if (noCam) noCam.style.display = 'none';

        deafCamera = new Camera(video, {
            onFrame: async () => {
                if (currentMode === 'deaf') {
                    if (awaitingHeadConfirmation) {
                        // When waiting for Nod/Shake response, feed frames to head tracker
                        if (faceDetector) {
                            try {
                                await faceDetector.send({ image: video });
                            } catch (e) {
                                processOpticalHeadMovement(video);
                            }
                        } else {
                            processOpticalHeadMovement(video);
                        }
                    } else if (handsDetector && isDeafDetecting) {
                        // When actively listening for hand signs, feed frames to hands detector
                        await handsDetector.send({ image: video });
                    }
                }
            },
            width: 1280,
            height: 720
        });
        deafCamera.start();
        isDeafCamOn = true;
        if (camBtn) camBtn.classList.add('active');
    } catch (err) {
        console.warn('Deaf camera permission denied:', err);
        if (noCam) noCam.style.display = 'flex';
        showToast('Camera access required for sign detection');
    }
}

function stopDeafCamera() {
    const video = document.getElementById('deaf-camera-feed');
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
    isDeafCamOn = false;
    const camBtn = document.getElementById('deaf-cam-btn');
    if (camBtn) camBtn.classList.remove('active');
}

function toggleDeafCamera() {
    if (isDeafCamOn) stopDeafCamera();
    else startDeafCamera();
}

function toggleDeafDetection() {
    isDeafDetecting = !isDeafDetecting;
    const btn = document.getElementById('deaf-detect-btn');
    if (btn) btn.classList.toggle('active', isDeafDetecting);
    showToast(isDeafDetecting ? 'AI Detection Enabled' : 'AI Detection Paused');
}

function addDeafHistoryItem(word, confidence) {
    const list = document.getElementById('deaf-translation-history');
    const empty = document.getElementById('deaf-translation-empty');
    const badge = document.getElementById('deaf-history-badge');
    if (!list) return;
    if (empty) empty.style.display = 'none';

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

    list.insertBefore(item, list.firstChild);
    deafHistoryItems.push(item);
    if (badge) badge.textContent = deafHistoryItems.length;
}

function clearDeafHistory() {
    deafHistoryItems.forEach(i => i.remove());
    deafHistoryItems = [];
    predictionWindow = [];
    lastHistoryWord = '';
    const empty = document.getElementById('deaf-translation-empty');
    const badge = document.getElementById('deaf-history-badge');
    if (empty) empty.style.display = 'flex';
    if (badge) badge.textContent = '0';
    showToast('History cleared');
}

function switchDeafTab(tab) {
    const histBtn = document.getElementById('deaf-tab-history-btn');
    const dictBtn = document.getElementById('deaf-tab-dict-btn');
    const histContent = document.getElementById('deaf-tab-history');
    const dictContent = document.getElementById('deaf-tab-dict');

    if (tab === 'history') {
        if (histBtn) histBtn.classList.add('active');
        if (dictBtn) dictBtn.classList.remove('active');
        if (histContent) histContent.style.display = 'flex';
        if (dictContent) dictContent.style.display = 'none';
    } else {
        if (dictBtn) dictBtn.classList.add('active');
        if (histBtn) histBtn.classList.remove('active');
        if (dictContent) dictContent.style.display = 'flex';
        if (histContent) histContent.style.display = 'none';
        renderUnifiedDictionary(allSignLabels);
    }
}

// ============== SIGNS DICTIONARY GUIDE ==============
async function fetchDatasetLabels() {
    try {
        const res = await fetch('/api/labels');
        const data = await res.json();
        allSignLabels = data.labels || [];
    } catch (err) {
        console.warn('Failed to load dataset labels:', err);
    }
}

// ============ SIGNS GUIDE MODAL HANDLERS ============
function openSignsModal() {
    const modal = document.getElementById('signs-modal');
    if (!modal) return;
    modal.style.display = 'flex';
    populateModalDictionary(allSignLabels);
    const searchInput = document.getElementById('modal-dict-search');
    if (searchInput) {
        searchInput.value = '';
        searchInput.focus();
    }
}

function closeSignsModal() {
    const modal = document.getElementById('signs-modal');
    if (modal) modal.style.display = 'none';
}

function filterModalDictionary(query) {
    if (!allSignLabels) return;
    const q = (query || '').toLowerCase().trim();
    if (!q) {
        populateModalDictionary(allSignLabels);
        return;
    }
    const filtered = allSignLabels.filter(item => {
        const name = (item.name || item).toLowerCase();
        return name.includes(q) || name.replace(/_/g, ' ').includes(q);
    });
    populateModalDictionary(filtered);
}

function populateModalDictionary(labels) {
    const container = document.getElementById('modal-dict-list');
    if (!container) return;
    if (!labels || labels.length === 0) {
        container.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);">No matching signs found</div>';
        return;
    }
    container.innerHTML = labels.map(item => {
        const name = typeof item === 'string' ? item : item.name;
        const count = item.image_count || 0;
        const imgUrl = item.image_url || `/dataset/${encodeURIComponent(name)}/1.jpg`;
        const displayName = name.replace(/_/g, ' ');
        return `
            <div class="dict-item" onclick="openSignModal('${name}')">
                <img class="dict-item-thumb" src="${imgUrl}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\' viewBox=\\'0 0 100 100\\'><text y=\\'.9em\\' font-size=\\'80\\'>🤟</text></svg>'" alt="${displayName}">
                <div style="flex:1;min-width:0;">
                    <div class="dict-item-name">${displayName}</div>
                    <div class="dict-item-sub">${count} frames · Click to view sample</div>
                </div>
            </div>
        `;
    }).join('');
}

function renderUnifiedDictionary(signs) {
    const list = document.getElementById('deaf-dict-list');
    if (!list) return;
    if (!signs || signs.length === 0) {
        list.innerHTML = '<div style="padding:16px;color:var(--text-muted);text-align:center;">No matching signs</div>';
        return;
    }
    list.innerHTML = signs.map(s => {
        const displayName = s.name.replace(/_/g, ' ');
        const thumbSrc = s.image_url || `/dataset/${encodeURIComponent(s.name)}/1.jpg`;
        return `
            <div class="dict-item" onclick="openSignModal('${s.name}')">
                <img class="dict-item-thumb" src="${thumbSrc}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\' viewBox=\\'0 0 100 100\\'><text y=\\'.9em\\' font-size=\\'80\\'>🤟</text></svg>'" alt="${displayName}">
                <div style="flex:1;min-width:0;">
                    <div class="dict-item-name">${displayName}</div>
                    <div class="dict-item-sub">${s.image_count} frames · Click to view sample</div>
                </div>
            </div>
        `;
    }).join('');
}

function filterUnifiedDictionary(query) {
    const q = query.toLowerCase().trim();
    if (!q) {
        renderUnifiedDictionary(allSignLabels);
    } else {
        const filtered = allSignLabels.filter(s =>
            s.name.toLowerCase().includes(q) || s.name.replace(/_/g, ' ').toLowerCase().includes(q)
        );
        renderUnifiedDictionary(filtered);
    }
}

async function openSignModal(signName) {
    try {
        const res = await fetch(`/api/get_sign/${encodeURIComponent(signName)}`);
        const data = await res.json();
        const displayName = signName.replace(/_/g, ' ');

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
    } catch (err) {
        showToast('Could not load sign preview');
    }
}

// ============== HEARING PERSON MODE (SPEECH & TEXT TO SIGN) ==============
function initSpeechEngine() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
        const mic = document.getElementById('normal-mic-btn');
        if (mic) {
            mic.style.opacity = '0.5';
            mic.title = 'Speech recognition not supported in this browser (Use Chrome or Edge)';
        }
        return;
    }

    speechRecognition = new SpeechRec();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;
    speechRecognition.lang = 'en-US';

    speechRecognition.onstart = () => {
        isListening = true;
        const ind = document.getElementById('speech-indicator');
        const micBtn = document.getElementById('normal-mic-btn');
        if (ind) ind.classList.add('active');
        if (micBtn) micBtn.classList.add('active');
        showToast('🎤 Listening... Speak now');
    };

    speechRecognition.onresult = (event) => {
        let interim = '';
        let final = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) final += transcript;
            else interim += transcript;
        }

        const textToShow = final || interim;
        const msgDisp = document.getElementById('normal-message-display');
        const textDisp = document.getElementById('normal-speech-text');
        if (msgDisp && textDisp && textToShow) {
            msgDisp.style.display = 'block';
            textDisp.textContent = `🗣️ "${textToShow}"`;
        }

        if (final) {
            processWordsToSigns(final, 'normal');
        }
    };

    speechRecognition.onerror = (e) => {
        console.warn('Speech error:', e.error);
        stopSpeechRecognition();
    };

    speechRecognition.onend = () => {
        isListening = false;
        const ind = document.getElementById('speech-indicator');
        const micBtn = document.getElementById('normal-mic-btn');
        if (ind) ind.classList.remove('active');
        if (micBtn) micBtn.classList.remove('active');
    };
}

function toggleSpeechRecognition() {
    if (!speechRecognition) {
        showToast('Speech recognition requires Chrome or Edge browser');
        return;
    }
    if (isListening) stopSpeechRecognition();
    else {
        try { speechRecognition.start(); }
        catch (err) { speechRecognition.stop(); }
    }
}

function stopSpeechRecognition() {
    if (speechRecognition && isListening) {
        speechRecognition.stop();
        isListening = false;
    }
}

function sendNormalText() {
    const input = document.getElementById('normal-text-input');
    const text = input ? input.value.trim() : '';
    if (!text) return;
    input.value = '';

    const msgDisp = document.getElementById('normal-message-display');
    const textDisp = document.getElementById('normal-speech-text');
    if (msgDisp && textDisp) {
        msgDisp.style.display = 'block';
        textDisp.textContent = `💬 "${text}"`;
    }

    processWordsToSigns(text, 'normal');
}

async function processWordsToSigns(text, targetMode) {
    try {
        const res = await fetch('/api/match_words', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await res.json();
        const matches = data.matches || [];

        if (targetMode === 'normal') {
            displaySignCards(matches);
        } else if (targetMode === 'dual') {
            displayDualSignCards(matches, text);
        }
    } catch (err) {
        console.error('Word match error:', err);
    }
}

function displaySignCards(matches) {
    const track = document.getElementById('normal-sign-track');
    const empty = document.getElementById('normal-sign-empty');
    const countBadge = document.getElementById('normal-sign-count');
    if (!track) return;

    if (!matches || matches.length === 0) {
        showToast('No matching signs found in dataset');
        return;
    }

    if (empty) empty.style.display = 'none';

    // Clear previous cards to display current phrase cleanly from left to right
    track.innerHTML = '';
    normalSignCards = [];

    matches.forEach((m, idx) => {
        const card = document.createElement('div');
        card.className = 'sign-sequence-card';
        card.innerHTML = `
            <div class="seq-num">${idx + 1}</div>
            <img class="seq-card-img" src="${m.image_url}" alt="${m.sign}" loading="lazy">
            <div class="seq-card-body">
                <div class="seq-card-word">🤟 ${m.sign.replace(/_/g, ' ')}</div>
                <div class="seq-card-meta">Word: "${m.word}" · 1st Frame</div>
            </div>
        `;
        // Append in left-to-right chronological order
        track.appendChild(card);
        normalSignCards.push(card);
    });

    if (countBadge) countBadge.textContent = `${matches.length} sign${matches.length > 1 ? 's' : ''}`;
    showToast(`Found ${matches.length} matching sign(s) in order!`);
    track.scrollLeft = 0;
}

function clearSignSequence() {
    const track = document.getElementById('normal-sign-track');
    const countBadge = document.getElementById('normal-sign-count');
    const msgDisp = document.getElementById('normal-message-display');
    const textDisp = document.getElementById('normal-speech-text');

    if (track) {
        track.innerHTML = `
            <div class="sign-seq-empty" id="normal-sign-empty">
                <span>💬 Speak into the microphone or type below — matched sign images will appear here in left-to-right sequence.</span>
            </div>
        `;
    }
    if (countBadge) countBadge.textContent = '0 signs';
    if (msgDisp) msgDisp.style.display = 'none';
    if (textDisp) textDisp.textContent = '';
    normalSignCards = [];
}

async function toggleNormalCamera() {
    const video = document.getElementById('normal-camera-feed');
    const placeholder = document.getElementById('normal-video-placeholder');
    const camBtn = document.getElementById('normal-cam-btn');

    if (isNormalCamOn) {
        stopNormalCamera();
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720, facingMode: 'user' },
                audio: false
            });
            if (video) {
                video.srcObject = stream;
                video.style.display = 'block';
            }
            if (placeholder) placeholder.style.display = 'none';
            isNormalCamOn = true;
            if (camBtn) camBtn.classList.add('active');
        } catch (err) {
            showToast('Camera permission denied');
        }
    }
}

function stopNormalCamera() {
    const video = document.getElementById('normal-camera-feed');
    const placeholder = document.getElementById('normal-video-placeholder');
    const camBtn = document.getElementById('normal-cam-btn');
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
        video.style.display = 'none';
    }
    if (placeholder) placeholder.style.display = 'flex';
    isNormalCamOn = false;
    if (camBtn) camBtn.classList.remove('active');
}

// ============== TWO-WAY MEET (DUAL VIEW) ==============
async function startDualCamera() {
    const video = document.getElementById('dual-camera-feed');
    const canvas = document.getElementById('dual-landmark-canvas');
    if (!video || !canvas) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 360, facingMode: 'user' },
            audio: false
        });
        video.srcObject = stream;
        await new Promise(r => { video.onloadedmetadata = r; });
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        dualCamera = new Camera(video, {
            onFrame: async () => {
                if (handsDetector && currentMode === 'dual') {
                    await handsDetector.send({ image: video });
                }
            },
            width: 640,
            height: 360
        });
        dualCamera.start();
    } catch (err) {
        console.warn('Dual camera error:', err);
    }
}

function stopDualCamera() {
    const video = document.getElementById('dual-camera-feed');
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
}

function toggleDualMic() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
        showToast('Speech recognition not supported');
        return;
    }
    const btn = document.getElementById('dual-mic-btn');
    const status = document.getElementById('dual-speech-status');

    if (isDualListening) {
        stopSpeechRecognition();
        isDualListening = false;
        if (btn) btn.classList.remove('active');
        if (status) status.textContent = 'Mic Off';
    } else {
        const dualRec = new SpeechRec();
        dualRec.lang = 'en-US';
        dualRec.onstart = () => {
            isDualListening = true;
            if (btn) btn.classList.add('active');
            if (status) status.textContent = '🎤 Listening...';
        };
        dualRec.onresult = (e) => {
            const text = e.results[0][0].transcript;
            sendDualTextExplicit(text);
        };
        dualRec.onend = () => {
            isDualListening = false;
            if (btn) btn.classList.remove('active');
            if (status) status.textContent = 'Mic Ready';
        };
        dualRec.start();
    }
}

function sendDualText() {
    const input = document.getElementById('dual-text-input');
    const text = input ? input.value.trim() : '';
    if (!text) return;
    input.value = '';
    sendDualTextExplicit(text);
}

function sendDualTextExplicit(text) {
    addDualDialogue('hearing', text);
    processWordsToSigns(text, 'dual');
}

function displayDualSignCards(matches, rawText) {
    const output = document.getElementById('dual-sign-output');
    const empty = document.getElementById('dual-sign-empty');
    if (!output) return;
    if (empty) empty.style.display = 'none';

    output.innerHTML = '';
    if (!matches || matches.length === 0) {
        output.innerHTML = `
            <div style="padding:16px;text-align:center;color:var(--text-secondary);">
                🗣️ "${rawText}"<br><small style="color:var(--text-muted);">(No exact sign image in dataset)</small>
            </div>
        `;
        return;
    }

    matches.forEach(m => {
        const card = document.createElement('div');
        card.style.cssText = 'display:inline-block;width:130px;margin:6px;background:rgba(255,255,255,0.06);border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.1);text-align:center;';
        card.innerHTML = `
            <img src="${m.image_url}" style="width:100%;height:80px;object-fit:cover;">
            <div style="padding:6px;font-size:12px;font-weight:700;">${m.sign.replace(/_/g, ' ')}</div>
        `;
        output.appendChild(card);
    });
}

function addDualDialogue(sender, message, confidence = null) {
    const list = document.getElementById('dual-dialogue-list');
    const empty = document.getElementById('dual-dialogue-empty');
    if (!list) return;
    if (empty) empty.style.display = 'none';

    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const item = document.createElement('div');
    item.style.cssText = 'padding:10px 14px;border-radius:12px;margin-bottom:8px;font-size:13px;animation:slide-in-right 0.25s ease-out;';

    if (sender === 'hearing') {
        item.style.background = 'rgba(99, 102, 241, 0.15)';
        item.style.border = '1px solid rgba(99, 102, 241, 0.3)';
        item.innerHTML = `
            <div style="font-weight:700;color:#a5b4fc;margin-bottom:2px;">🗣️ Hearing Companion <span style="float:right;font-size:11px;color:var(--text-muted);">${timeStr}</span></div>
            <div>${message}</div>
        `;
    } else {
        item.style.background = 'rgba(236, 72, 153, 0.15)';
        item.style.border = '1px solid rgba(236, 72, 153, 0.3)';
        const pct = confidence ? ` (${(confidence * 100).toFixed(0)}%)` : '';
        item.innerHTML = `
            <div style="font-weight:700;color:#f472b6;margin-bottom:2px;">🤟 Deaf Companion <span style="float:right;font-size:11px;color:var(--text-muted);">${timeStr}</span></div>
            <div><strong>"${message}"</strong>${pct}</div>
        `;
    }

    list.insertBefore(item, list.firstChild);
    dualDialogueItems.push(item);
}

function clearDualDialogue() {
    dualDialogueItems.forEach(i => i.remove());
    dualDialogueItems = [];
    const empty = document.getElementById('dual-dialogue-empty');
    if (empty) empty.style.display = 'flex';
    showToast('Dialogue cleared');
}

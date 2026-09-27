/**
 * HAND SLINGER - Normal (Hearing) Person Mode
 * =============================================
 * Google Meet-like interface with:
 * - Web Speech API for voice recognition
 * - Text input for manual typing
 * - Sign language image display from dataset
 * - Camera feed (decorative, simulates video call)
 */

// ============== DOM ELEMENTS ==============
const cameraFeed = document.getElementById('camera-feed');
const videoPlaceholder = document.getElementById('video-placeholder');
const speechIndicator = document.getElementById('speech-indicator');
const messageDisplay = document.getElementById('message-display');
const speechTextDisplay = document.getElementById('speech-text-display');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const micIcon = document.getElementById('mic-icon');
const camBtn = document.getElementById('cam-btn');
const camIcon = document.getElementById('cam-icon');
const endBtn = document.getElementById('end-btn');
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const signPanelContent = document.getElementById('sign-panel-content');
const signPanelEmpty = document.getElementById('sign-panel-empty');
const signCount = document.getElementById('sign-count');
const toast = document.getElementById('toast');

// ============== STATE ==============
let stream = null;
let recognition = null;
let isListening = false;
let isCameraOn = false;
let signCards = [];
let processedWords = new Set();

// ============== TOAST NOTIFICATION ==============
function showToast(message, duration = 3000) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}

// ============== CAMERA ==============
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: 'user' },
            audio: false
        });
        cameraFeed.srcObject = stream;
        videoPlaceholder.style.display = 'none';
        cameraFeed.style.display = 'block';
        isCameraOn = true;
        camIcon.textContent = '📷';
        camBtn.classList.add('active');
    } catch (err) {
        console.warn('Camera access denied:', err);
        showToast('Camera access denied. You can still use voice & text input.');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraFeed.srcObject = null;
    cameraFeed.style.display = 'none';
    videoPlaceholder.style.display = 'flex';
    isCameraOn = false;
    camIcon.textContent = '📷';
    camBtn.classList.remove('active');
}

// ============== SPEECH RECOGNITION ==============
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        showToast('Speech recognition not supported. Please use Chrome or Edge.');
        micBtn.style.opacity = '0.5';
        micBtn.style.pointerEvents = 'none';
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        isListening = true;
        speechIndicator.classList.add('active');
        statusBadge.classList.add('recording');
        statusText.textContent = 'Listening...';
        micBtn.classList.add('active');
        micIcon.textContent = '🎤';
    };

    recognition.onend = () => {
        isListening = false;
        speechIndicator.classList.remove('active');
        statusBadge.classList.remove('recording');
        statusText.textContent = 'Ready';
        micBtn.classList.remove('active');
        micIcon.textContent = '🎤';

        // Don't hide the last message immediately
    };

    recognition.onresult = (event) => {
        let interimText = '';
        let finalText = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript.trim();
            if (event.results[i].isFinal) {
                finalText += transcript + ' ';
            } else {
                interimText += transcript;
            }
        }

        // Show interim/final text as overlay
        const displayText = finalText || interimText;
        if (displayText) {
            messageDisplay.style.display = 'block';
            speechTextDisplay.textContent = displayText;
        }

        // Process final text for sign translation
        if (finalText.trim()) {
            processTextForSigns(finalText.trim());
        }
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'not-allowed') {
            showToast('Microphone access denied. Please grant permission.');
        } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
            showToast(`Speech error: ${event.error}`);
        }
    };
}

function toggleMic() {
    if (!recognition) {
        showToast('Speech recognition not available');
        return;
    }

    if (isListening) {
        recognition.stop();
    } else {
        try {
            recognition.start();
        } catch (e) {
            // Already started, restart
            recognition.stop();
            setTimeout(() => recognition.start(), 100);
        }
    }
}

// ============== SIGN TRANSLATION ==============
async function processTextForSigns(text) {
    try {
        const response = await fetch('/api/match_words', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) return;

        const data = await response.json();

        if (data.matches && data.matches.length > 0) {
            data.matches.forEach(match => {
                // Only add if not already shown (avoid duplicates)
                if (!processedWords.has(match.sign)) {
                    processedWords.add(match.sign);
                    addSignCard(match);
                }
            });
        }
    } catch (err) {
        console.error('Error matching words:', err);
    }
}

async function processTypedText(text) {
    if (!text.trim()) return;

    // Show text overlay
    messageDisplay.style.display = 'block';
    speechTextDisplay.textContent = text;

    // Clear after 4 seconds
    setTimeout(() => {
        messageDisplay.style.display = 'none';
    }, 4000);

    await processTextForSigns(text);
}

function addSignCard(match) {
    // Remove empty state
    if (signPanelEmpty) {
        signPanelEmpty.style.display = 'none';
    }

    // Create sign card
    const card = document.createElement('div');
    card.className = 'sign-card';
    card.innerHTML = `
        <img class="sign-card-image" src="${match.image_url}" alt="Sign for ${match.sign}"
             onerror="this.style.display='none'">
        <div class="sign-card-info">
            <div class="sign-card-word">${match.sign.replace(/_/g, ' ')}</div>
            <div class="sign-card-meta">Matched: "${match.word}" • ${match.total_frames} frames</div>
        </div>
    `;

    // Prepend (newest first)
    signPanelContent.insertBefore(card, signPanelContent.firstChild);
    signCards.push(card);

    // Update count
    signCount.textContent = signCards.length;

    // Scroll to top
    signPanelContent.scrollTop = 0;
}

// ============== TEXT INPUT HANDLING ==============
function handleSend() {
    const text = textInput.value.trim();
    if (!text) return;

    processTypedText(text);
    textInput.value = '';
    textInput.focus();
}

// ============== EVENT LISTENERS ==============

// Mic toggle
micBtn.addEventListener('click', toggleMic);

// Camera toggle
camBtn.addEventListener('click', () => {
    if (isCameraOn) {
        stopCamera();
    } else {
        startCamera();
    }
});

// Send button
sendBtn.addEventListener('click', handleSend);

// Enter key in text input
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        handleSend();
    }
});

// End call / Leave
endBtn.addEventListener('click', () => {
    if (recognition) recognition.stop();
    stopCamera();
    window.location.href = '/';
});

// ============== INITIALIZATION ==============
document.addEventListener('DOMContentLoaded', () => {
    initSpeechRecognition();
    startCamera();
    statusText.textContent = 'Ready';
    showToast('Welcome! Speak or type to translate to sign language.');
});

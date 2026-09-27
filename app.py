"""
HAND SLINGER - Flask Web Application
=====================================
Two-way communication platform between hearing and deaf persons.
- Normal mode: Speech/text → Sign language images
- Deaf mode: Camera hand signs → Text translation

API Endpoints:
  /api/predict      - Predict sign from MediaPipe landmarks
  /api/get_sign     - Get first sign image for a word
  /api/match_words  - Match text to available sign classes
  /api/labels       - List all available sign labels
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
import os
import re
import joblib
import json
import numpy as np

app = Flask(__name__)

# ============== CONFIGURATION ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Frames_Word_Level")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Global state
model = None
label_encoder = None
class_names = []
WORD_MAPPING = {}


def compute_extra_features(landmarks_126):
    """
    Same feature engineering as training - must stay in sync with train_model.py.
    Computes 26 extra features: distances, curl ratios, angles, palm geometry.
    """
    lm = np.array(landmarks_126[:63]).reshape(21, 3)
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    extra = []

    # 1. Pairwise fingertip distances (10)
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dist = np.sqrt(np.sum((lm[tips[i]] - lm[tips[j]]) ** 2))
            extra.append(dist)

    # 2. Finger curl ratios (5)
    for tip, mcp in zip(tips, mcps):
        tip_dist = np.sqrt(np.sum(lm[tip] ** 2))
        mcp_dist = np.sqrt(np.sum(lm[mcp] ** 2))
        curl = tip_dist / (mcp_dist + 1e-6)
        extra.append(curl)

    # 3. Thumb to each fingertip distance (4)
    for tip in tips[1:]:
        dist = np.sqrt(np.sum((lm[4] - lm[tip]) ** 2))
        extra.append(dist)

    # 4. Finger spread angles (4)
    for i in range(len(tips) - 1):
        v1 = lm[tips[i]] - lm[0]
        v2 = lm[tips[i+1]] - lm[0]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        extra.append(cos_angle)

    # 5. Palm geometry (3)
    palm_width = np.sqrt(np.sum((lm[5] - lm[17]) ** 2))
    extra.append(palm_width)
    palm_height = np.sqrt(np.sum(lm[9] ** 2))
    extra.append(palm_height)
    extra.append(palm_width / (palm_height + 1e-6))

    return extra


def load_model():
    """Load the trained model and label encoder."""
    global model, label_encoder, class_names

    model_path = os.path.join(MODEL_DIR, "hand_sign_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    class_names_path = os.path.join(MODEL_DIR, "class_names.json")

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
        else:
            class_names = list(label_encoder.classes_)
        print(f"[OK] Model loaded - {len(class_names)} sign classes available")
    else:
        print("[WARNING] No trained model found. Run 'python train_model.py' first!")
        # Populate class_names from dataset folders
        if os.path.exists(DATASET_PATH):
            class_names = sorted([d for d in os.listdir(DATASET_PATH)
                                  if os.path.isdir(os.path.join(DATASET_PATH, d))])
            print(f"[INFO] Found {len(class_names)} classes in dataset (model not trained yet)")


def build_word_mapping():
    """Build a comprehensive mapping from common words/phrases → dataset class names."""
    global WORD_MAPPING

    for class_name in class_names:
        # Exact class name (lowercase)
        WORD_MAPPING[class_name.lower()] = class_name

        # Replace underscores with spaces
        clean = class_name.replace('_', ' ').lower()
        WORD_MAPPING[clean] = class_name

        # Each individual word in the class name
        parts = class_name.replace('_', ' ').replace("'", "").split()
        for part in parts:
            part_lower = part.lower()
            if len(part_lower) > 1:  # Skip single-char words (except 'i')
                WORD_MAPPING[part_lower] = class_name

    # Special mappings for compound names
    special_maps = {
        "I_ME_MINE_MY": ["i", "me", "mine", "my", "myself", "i'm", "im"],
        "LIKE_LOVE": ["like", "love", "loving", "loved"],
        "HELLO_HI": ["hello", "hi", "hey", "greetings"],
        "COLLEGE_SCHOOL": ["college", "school", "university", "campus"],
        "OLD_AGE": ["old", "age", "elderly", "aged"],
        "DON'T CARE": ["don't care", "dont care", "don't care", "careless"],
        "A LOT": ["a lot", "alot", "lot", "lots", "many", "much"],
        "THANK": ["thank", "thanks", "thank you", "thankyou", "grateful"],
        "SORRY": ["sorry", "apologize", "apology", "apologies"],
        "PLEASE": ["please", "pls", "plz", "kindly"],
        "GOOD": ["good", "great", "well", "fine", "nice"],
        "BAD": ["bad", "terrible", "awful", "poor"],
        "HAPPY": ["happy", "happiness", "glad", "joyful", "joy"],
        "ANGRY": ["angry", "mad", "furious", "anger"],
        "AFRAID": ["afraid", "scared", "fear", "frightened"],
        "HUNGRY": ["hungry", "hunger", "starving"],
        "THIRSTY": ["thirsty", "thirst"],
        "TIRED": ["tired", "exhausted", "fatigue"],
        "FRIEND": ["friend", "buddy", "pal", "companion"],
        "FOOD": ["food", "eat", "eating", "meal"],
        "WATER": ["water", "drink", "drinking"],
        "HELP": ["help", "helping", "assist", "assistance"],
        "STOP": ["stop", "halt", "cease", "quit"],
        "COME": ["come", "coming", "arrive"],
        "GO": ["go", "going", "leave", "went"],
        "WANT": ["want", "wanted", "need", "desire"],
        "HEAR": ["hear", "hearing", "listen", "listening"],
        "SPEAK": ["speak", "speaking", "say", "saying"],
        "UNDERSTAND": ["understand", "understood", "comprehend"],
        "MEET": ["meet", "meeting", "encounter"],
        "SLEEP": ["sleep", "sleeping", "asleep", "nap"],
        "NAME": ["name", "called", "named"],
        "WHERE": ["where", "location", "place"],
        "WHAT": ["what", "which"],
        "HOW": ["how"],
        "WHO": ["who", "whom", "whose"],
        "WELCOME": ["welcome"],
        "CONGRATULATIONS": ["congratulations", "congrats", "congratulation"],
        "TAKE CARE": ["take care", "careful", "care"],
        "ON THE WAY": ["on the way", "coming", "otw"],
        "SOME ONE": ["someone", "somebody", "some one"],
        "SOMETHING": ["something", "thing"],
        "SOME HOW": ["somehow", "some how"],
        "SO MUCH": ["so much"],
        "MEAN IT": ["mean it", "serious", "mean"],
        "TURN ON": ["turn on", "switch on"],
        "TAKE TIME": ["take time", "patience"],
        "TALK": ["talk", "talking", "chat", "chatting"],
        "TELL": ["tell", "told", "telling"],
        "THINK": ["think", "thinking", "thought"],
        "TRUST": ["trust", "trusting", "trusted"],
        "TRUTH": ["truth", "true", "honest"],
        "PROMISE": ["promise", "swear", "vow"],
        "REPEAT": ["repeat", "again", "once more"],
        "REALLY": ["really", "truly", "seriously"],
        "ENJOY": ["enjoy", "enjoying", "fun"],
        "CRYING": ["crying", "cry", "tears", "weeping"],
        "BORED": ["bored", "boring", "boredom"],
        "COLD": ["cold", "chilly", "freezing"],
        "FEVER": ["fever", "sick", "ill"],
        "MEDICINE": ["medicine", "medication", "drug", "pill"],
        "PHONE": ["phone", "call", "telephone", "mobile"],
        "NUMBER": ["number", "digit", "count"],
        "ROOM": ["room", "bedroom", "place"],
        "SHIRT": ["shirt", "clothes", "clothing"],
        "BED": ["bed", "bedroom"],
        "PREPARE": ["prepare", "ready", "preparation"],
        "BRING": ["bring", "brought", "fetch"],
        "SERVE": ["serve", "serving", "service"],
        "NICE": ["nice", "pleasant", "lovely"],
        "KIND": ["kind", "kindness", "generous"],
        "BEAUTIFUL": ["beautiful", "pretty", "gorgeous"],
        "AGREE": ["agree", "agreed", "agreement"],
        "LIKE": ["like", "liked"],
        "FINE": ["fine", "okay", "ok", "alright"],
        "FREE": ["free", "freedom"],
        "ABUSE": ["abuse", "abusive"],
        "DARE": ["dare", "daring", "challenge"],
        "GRATEFUL": ["grateful", "thankful", "appreciation"],
        "APPRECIATE": ["appreciate", "appreciation", "valued"],
        "DISAPPOINTED": ["disappointed", "disappointment", "letdown"],
        "STUBBORN": ["stubborn", "obstinate"],
        "DILEMMA": ["dilemma", "confused", "confusion"],
        "DIFFERENCE": ["difference", "different", "differ"],
        "HIDING": ["hiding", "hide", "hidden"],
        "HURT": ["hurt", "pain", "painful", "ache"],
        "FAVOUR": ["favour", "favor", "request"],
        "HAPPENED": ["happened", "happen", "occurs", "occurred"],
        "HEART": ["heart", "love", "soul"],
        "LEAVE": ["leave", "leaving", "left", "depart"],
        "OUTSIDE": ["outside", "outdoor", "out"],
        "POUR": ["pour", "pouring", "spill"],
        "SITTING": ["sitting", "sit", "seated"],
        "SLOWER": ["slower", "slow", "slowly"],
        "SOFTLY": ["softly", "soft", "gentle", "gently"],
        "SURE": ["sure", "certain", "definitely"],
        "THINGS": ["things", "stuff", "items"],
        "TODAY": ["today", "this day"],
        "TRAIN": ["train", "training", "practice"],
        "WEAR": ["wear", "wearing", "put on"],
        "WORRY": ["worry", "worried", "worrying", "anxious"],
        "YOU": ["you", "your", "yours", "yourself"],
        "NOT": ["not", "no", "nope", "don't", "dont"],
        "ALL": ["all", "every", "everything", "everyone"],
        "ANYTHING": ["anything", "any"],
        "BECOME": ["become", "became", "becoming"],
        "COMB": ["comb", "brush", "hair"],
        "CLASS": ["class", "classroom", "lecture"],
        "DO": ["do", "doing", "does", "did", "done"],
        "FROM": ["from"],
        "HAD": ["had", "have", "has"],
        "PLACE": ["place", "location", "spot"],
        "THAT": ["that", "this", "those"],
    }

    for class_name, words in special_maps.items():
        if class_name in class_names:
            for w in words:
                WORD_MAPPING[w.lower()] = class_name


# ============== ROUTES ==============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/normal')
def normal_mode():
    return redirect('/?mode=normal')


@app.route('/deaf')
def deaf_mode():
    return redirect('/?mode=deaf')


# ============== API ENDPOINTS ==============

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict hand sign from MediaPipe landmark data with multi-view invariance (handles left/right hand and camera mirroring)."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 500

    data = request.get_json()
    raw_landmarks = data.get('landmarks', [])

    if not raw_landmarks or len(raw_landmarks) < 63:
        return jsonify({'error': 'Invalid landmark data'}), 400

    # Ensure 126 coords (pad if single hand)
    lm = list(raw_landmarks[:126])
    while len(lm) < 126:
        lm.append(0.0)

    # Candidate 1: Direct coordinates
    candidates = [lm]

    # Candidate 2: Horizontally flipped (x -> -x) for opposite hand or mirrored camera
    flipped = list(lm)
    for k in range(0, 126, 3):
        flipped[k] = -flipped[k]
    candidates.append(flipped)

    # If two hands detected (hand 2 coordinates not all zero), also evaluate swapped hands
    has_hand_2 = any(abs(v) > 1e-4 for v in lm[63:126])
    if has_hand_2:
        swapped = lm[63:126] + lm[:63]
        candidates.append(swapped)
        flipped_swapped = list(swapped)
        for k in range(0, 126, 3):
            flipped_swapped[k] = -flipped_swapped[k]
        candidates.append(flipped_swapped)

    # Vectorized batch evaluation across all views
    features_batch = []
    for cand in candidates:
        extra = compute_extra_features(cand)
        features_batch.append(cand + extra)

    all_probs = model.predict_proba(np.array(features_batch))

    # Pick candidate with highest top prediction confidence
    best_candidate_idx = int(np.argmax([np.max(p) for p in all_probs]))
    best_probs = all_probs[best_candidate_idx]

    pred_idx = int(np.argmax(best_probs))
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(best_probs[pred_idx])

    # Top 3 predictions
    top_3_idx = np.argsort(best_probs)[-3:][::-1]
    top_3 = []
    for idx in top_3_idx:
        top_3.append({
            'label': label_encoder.inverse_transform([idx])[0],
            'confidence': round(float(best_probs[idx]), 4)
        })

    return jsonify({
        'prediction': predicted_label,
        'confidence': round(confidence, 4),
        'top_3': top_3
    })


@app.route('/api/get_sign/<word>')
def get_sign(word):
    """Get the first sign image for a given word."""
    word_lower = word.lower().strip()

    # Try exact mapping first
    class_name = WORD_MAPPING.get(word_lower)

    # Try partial matching
    if not class_name:
        for key, value in WORD_MAPPING.items():
            if word_lower in key or key in word_lower:
                class_name = value
                break

    if not class_name:
        return jsonify({'error': f'No sign found for "{word}"', 'word': word}), 404

    # Get first image
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.exists(class_path):
        return jsonify({'error': f'Dataset folder missing for "{class_name}"'}), 404

    images = sorted([f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    if not images:
        return jsonify({'error': f'No images found for "{class_name}"'}), 404

    return jsonify({
        'word': class_name,
        'original_query': word,
        'image_url': f'/dataset/{class_name}/{images[0]}',
        'total_frames': len(images)
    })


@app.route('/api/match_words', methods=['POST'])
def match_words():
    """Match spoken/typed text to available sign classes strictly in left-to-right order."""
    data = request.get_json()
    text = data.get('text', '').lower().strip()

    if not text:
        return jsonify({'matches': []})

    # Clean punctuation to allow clean tokenization
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    words = [w for w in cleaned.split() if w]

    matches = []
    i = 0
    # Process from left to right across words in exact sentence order
    while i < len(words):
        matched = False
        # Try matching up to 4 consecutive words starting at index i (longest match first)
        max_len = min(4, len(words) - i)
        for length in range(max_len, 0, -1):
            phrase = ' '.join(words[i:i + length])
            class_name = WORD_MAPPING.get(phrase)
            if class_name:
                class_path = os.path.join(DATASET_PATH, class_name)
                if os.path.exists(class_path):
                    images = sorted([f for f in os.listdir(class_path)
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    if images:
                        matches.append({
                            'word': phrase,
                            'sign': class_name,
                            'image_url': f'/dataset/{class_name}/{images[0]}',
                            'total_frames': len(images),
                            'index': len(matches) + 1
                        })
                        i += length
                        matched = True
                        break
        if not matched:
            i += 1

    return jsonify({'matches': matches, 'input_text': text})


@app.route('/api/labels')
def get_labels():
    """Get all available sign language labels with thumbnail URLs."""
    labels_info = []
    for name in class_names:
        class_path = os.path.join(DATASET_PATH, name)
        count = 0
        image_url = ""
        if os.path.exists(class_path):
            imgs = sorted([f for f in os.listdir(class_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            count = len(imgs)
            if imgs:
                image_url = f"/dataset/{name}/{imgs[0]}"
        labels_info.append({'name': name, 'image_count': count, 'image_url': image_url})

    return jsonify({
        'labels': labels_info,
        'total_classes': len(class_names),
        'model_loaded': model is not None
    })


@app.route('/dataset/<path:filepath>')
def serve_dataset_image(filepath):
    """Serve images from the dataset directory."""
    return send_from_directory(DATASET_PATH, filepath)


# ============== MAIN ==============

if __name__ == '__main__':
    load_model()
    build_word_mapping()
    print(f"\n[INFO] Word mapping built: {len(WORD_MAPPING)} entries")
    print(f"[INFO] Starting Hand Slinger server...")
    print(f"[INFO] Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

import cv2
import numpy as np

CHARACTERS = ["blank", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", " ", " "]


def preprocess(image, width=320, height=48, pad_value=128):
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def decode(output):
    logits = np.asarray(output)
    logits = logits.squeeze(axis=0) if logits.ndim == 3 and logits.shape[0] == 1 else logits
    if logits.ndim != 2: raise ValueError(f"Unexpected recognizer output shape: {np.asarray(output).shape}")
    indices, confidence = logits.argmax(axis=1), logits.max(axis=1)
    text, scores, previous = [], [], -1
    for index, score in zip(indices, confidence):
        index = int(index)
        if index != 0 and index != previous and index < len(CHARACTERS): text.append(CHARACTERS[index]); scores.append(float(score))
        previous = index
    return "".join(text), float(np.mean(scores)) if scores else 0.0

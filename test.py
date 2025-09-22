import os
import random
import numpy as np
import tensorflow as tf
import gdown
import editdistance
import cv2
from typing import List
import json

# Enforce deterministic behavior
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from modelutil import load_model
from utils import num_to_char, char_to_num, load_data, load_video, load_alignments

# Load model architecture
model = load_model()

# Load weights from the HDF5 file
model.load_weights('checkpoints.weights.h5')

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER)."""
    return editdistance.eval(reference, hypothesis) / max(len(reference), 1)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return editdistance.eval(ref_words, hyp_words) / max(len(ref_words), 1)

# === SINGLE VIDEO EVALUATION ===



print("\n🔹 Evaluating on a SELECTED SINGLE FILE...\n")

file_path = './data/s1/bbas3a.mpg'



# Load and prepare data
sample = load_data(tf.convert_to_tensor(file_path))
real_chars = tf.strings.reduce_join([num_to_char(char) for char in sample[1]]).numpy().decode('utf-8')
print(' REAL TEXT:', real_chars)

# Predict (set training=False to avoid any Dropout or similar randomness)
yhat = model.predict(tf.expand_dims(sample[0], axis=0), verbose=0)
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
pred_chars = tf.strings.reduce_join([num_to_char(char) for char in decoded[0]]).numpy().decode('utf-8')

print(' PREDICTED TEXT:', pred_chars)

# Accuracy for single video
edit_dist = editdistance.eval(real_chars, pred_chars)
char_accuracy = (1 - edit_dist / max(len(real_chars), 1)) * 100
print(f" Character Accuracy (Single File): {char_accuracy:.2f}%")

real_words = real_chars.split()
pred_words = pred_chars.split()
word_dist = editdistance.eval(real_words, pred_words)
word_accuracy = (1 - word_dist / max(len(real_words), 1)) * 100
print(f" Word Accuracy (Single File): {word_accuracy:.2f}%")

cer = calculate_cer(real_chars, pred_chars) * 100
wer = calculate_wer(real_chars, pred_chars) * 100

print(f" Character Error Rate (Single File): {cer:.2f}%")
print(f" Word Error Rate (Single File): {wer:.2f}%")

# === FULL TEST SET EVALUATION ===

# Mapping function for dataset
def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

# Prepare dataset
data = tf.data.Dataset.list_files('./data/s1/*.mpg', shuffle=False)
data = data.shuffle(500, seed=SEED, reshuffle_each_iteration=False)
data = data.map(mappable_function, num_parallel_calls=1)  # avoid parallel nondeterminism
data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

# # Split dataset
train = data.take(450)
test = data.skip(450)

# Run test evaluation




test_data = test.as_numpy_iterator()

total_char_edit_dist = 0
total_char_len = 0
total_word_edit_dist = 0
total_word_len = 0
total_cer = 0
total_wer = 0
num_samples = 0

print("\nRunning predictions on full test set...\n")

for batch in test_data:
    videos, labels = batch

    # Predict deterministically
    yhat = model.predict(videos, verbose=0)
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75] * yhat.shape[0], greedy=True)[0][0].numpy()

    for i in range(len(labels)):
        real = tf.strings.reduce_join([num_to_char(c) for c in labels[i] if c != -1]).numpy().decode('utf-8')
        pred = tf.strings.reduce_join([num_to_char(c) for c in decoded[i] if c != -1]).numpy().decode('utf-8')

        # Character-level
        total_char_edit_dist += editdistance.eval(real, pred)
        total_char_len += len(real)

        # Word-level
        real_words = real.split()
        pred_words = pred.split()
        total_word_edit_dist += editdistance.eval(real_words, pred_words)
        total_word_len += len(real_words)

        # Error Rates
        total_cer += calculate_cer(real, pred)
        total_wer += calculate_wer(real, pred)
        num_samples += 1



char_accuracy = (1 - total_char_edit_dist / max(total_char_len, 1)) * 100
word_accuracy = (1 - total_word_edit_dist / max(total_word_len, 1)) * 100
avg_cer = (total_cer / max(num_samples, 1)) * 100
avg_wer = (total_wer / max(num_samples, 1)) * 100

print("\n" + "=" * 50)
print(f"Full Test Set Character Accuracy: {char_accuracy:.2f}%")
print(f"Full Test Set Word Accuracy: {word_accuracy:.2f}%")
print(f"Full Test Set Character Error Rate (CER): {avg_cer:.2f}%")
print(f"Full Test Set Word Error Rate (WER): {avg_wer:.2f}%")
print("=" * 50)



results = {
    "character_accuracy": round(char_accuracy, 2),
    "word_accuracy": round(word_accuracy, 2),
    "character_error_rate": round(avg_cer, 2),
    "word_error_rate": round(avg_wer, 2)
}

with open("accuracy_cache.json", "w") as f:
    json.dump(results, f, indent=2)

with open("accuracy_result.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n Accuracy results written to 'accuracy_cache.json' and 'accuracy_result.json'.")
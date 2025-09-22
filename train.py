# ------------------------- IMPORTS -------------------------
import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import gdown

# ------------------------- DOWNLOAD AND EXTRACT DATASET -------------------------
# URL of the dataset stored on Google Drive
url = 'https://drive.google.com/uc?id=1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK&confirm=t'
output = 'data.zip'

# Download and extract the dataset
# gdown.download(url, output, quiet=False)from typing import List
gdown.extractall('data.zip')


# ------------------------- LOAD VIDEO FUNCTION -------------------------
def load_video(path: str) -> List[float]:
    """
    Loads a video, converts each frame to grayscale, crops it,
    normalizes the frames, and returns a list of processed frames.
    """
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        # Crop the frame: height 190–236 and width 80–220
        frames.append(frame[190:236, 80:220, :])
    
    cap.release()

    # Normalize the frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


# ------------------------- CHARACTER MAPPINGS -------------------------
# Define the vocabulary used in alignment files
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create mappings from characters to numbers and vice versa
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


# ------------------------- LOAD ALIGNMENT FUNCTION -------------------------
def load_alignments(path: str) -> List[str]:
    """
    Loads the alignment file, filters out silence ('sil'),
    and converts the remaining phonemes into numerical form.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    
    # Convert to numerical format (excluding the first empty token)
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


# ------------------------- LOAD DATA FUNCTION -------------------------
def load_data(path: str):
    """
    Given the path to an alignment file, loads the corresponding video and alignment.
    """
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]

    # Construct full paths for video and alignment files
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    # Load frames and alignments
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments


# ------------------------- WRAPPER FOR TENSORFLOW DATA PIPELINE -------------------------
def mappable_function(path: str) -> List[str]:
    """
    Wraps the data loading function for use in TensorFlow data pipeline.
    """
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


# ------------------------- BUILD DATASET PIPELINE -------------------------
# Create dataset from video file paths
data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

# Split into training and test datasets
train = data.take(450)
test = data.skip(450)


# ------------------------- BUILD MODEL -------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# Define the sequential model
model = Sequential()

# 3D Convolutional layers for spatio-temporal feature extraction
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

# Reshape output for RNN layers into one dimention
model.add(Reshape((-1, 75 * 5 * 17)))

# Bi-directional LSTM layers
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))

# Output layer with softmax for character classification
model.add(Dense(char_to_num.vocabulary_size() + 1, activation='softmax'))

# Print model summary
model.summary()


# ------------------------- CUSTOM CTC LOSS FUNCTION -------------------------
def CTCLoss(y_true, y_pred):
    """
    Calculates the CTC (Connectionist Temporal Classification) loss.
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# ------------------------- LEARNING RATE SCHEDULER -------------------------
def scheduler(epoch, lr):
    """
    Custom learning rate scheduler that exponentially decays the learning rate after 30 epochs.
    """
    if epoch < 30:
        return float(lr)
    else:
        return lr * tf.math.exp(-0.1)

# ------------------------- COMPILE AND TRAIN MODEL -------------------------
# Compile model with Adam optimizer and custom CTC loss
model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# Create callbacks for saving model weights and adjusting learning rate
checkpoint_callback = ModelCheckpoint(
    os.path.join('models', 'checkpoint.weights.h5'),
    monitor='loss',
    save_weights_only=True
)
schedule_callback = LearningRateScheduler(scheduler)

# Train the model
model.fit(
    train,
    validation_data=test,
    epochs=100,
    callbacks=[checkpoint_callback, schedule_callback]
)

















































# import os
# import cv2
# import tensorflow as tf
# import numpy as np
# from typing import List
# import gdown

# # Download and extract dataset
# #google drive uploaded dataset 
# url='https://drive.google.com/uc?id=1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK&confirm=t'

# output = 'data.zip'
# gdown.download(url, output, quiet=False)
# gdown.extractall('data.zip')

# # Data loading utilities
# def load_video(path: str) -> List[float]: 
#     cap = cv2.VideoCapture(path)
#     frames = []
#     for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
#         ret, frame = cap.read()
#         frame = tf.image.rgb_to_grayscale(frame)
#         frames.append(frame[190:236, 80:220, :])
#     cap.release()
#     mean = tf.math.reduce_mean(frames)
#     std = tf.math.reduce_std(tf.cast(frames, tf.float32))
#     return tf.cast((frames - mean), tf.float32) / std

# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# def load_alignments(path: str) -> List[str]: 
#     with open(path, 'r') as f: 
#         lines = f.readlines() 
#     tokens = []
#     for line in lines:
#         line = line.split()
#         if line[2] != 'sil': 
#             tokens = [*tokens, ' ', line[2]]
#     return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# def load_data(path: str): 
#     path = bytes.decode(path.numpy())
#     file_name = path.split('/')[-1].split('.')[0]
#     video_path = os.path.join('data', 's1', f'{file_name}.mpg')
#     alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
#     frames = load_video(video_path) 
#     alignments = load_alignments(alignment_path)
#     return frames, alignments

# def mappable_function(path: str) -> List[str]:
#     result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
#     return result

# # Data pipeline
# data = tf.data.Dataset.list_files('./data/s1/*.mpg')
# data = data.shuffle(500, reshuffle_each_iteration=False)
# data = data.map(mappable_function)
# data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
# data = data.prefetch(tf.data.AUTOTUNE)

# train = data.take(450)
# test = data.skip(450)

# # Model definition
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# model = Sequential()
# model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool3D((1, 2, 2)))
# model.add(Conv3D(256, 3, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool3D((1, 2, 2)))
# model.add(Conv3D(75, 3, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool3D((1, 2, 2)))
# model.add(Reshape((-1, 75 * 5 * 17)))
# model.add(Bidirectional(LSTM(128, return_sequences=True)))
# model.add(Dropout(0.5))
# model.add(Bidirectional(LSTM(128, return_sequences=True)))
# model.add(Dropout(0.5))
# model.add(Dense(char_to_num.vocabulary_size() + 1, activation='softmax'))
# model.summary()

# # Custom loss
# def CTCLoss(y_true, y_pred):
#     batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#     input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#     label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
#     input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# # Learning rate scheduler
# def scheduler(epoch, lr):
#     if epoch < 30:
#         return float(lr)
#     else:
#         return lr * tf.math.exp(-0.1)

# # Compile and train
# model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
# checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint.weights.h5'), monitor='loss', save_weights_only=True)
# schedule_callback = LearningRateScheduler(scheduler)

# model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback])

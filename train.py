import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Configurtaions
Model_Saved = 'Models/Speech_Model' # Model Weights Save Path
TRAIN_ENABLED = 0 # Training Models
TRAIN_SET_SIZE = 95000 # Splitting training set and validation set
BATCH_SIZE = 64
EPOCHS = 20

# Set seed for experiment reproducibility
seed = 41
tf.random.set_seed(seed)
np.random.seed(seed)

# download 'Speech_commands' data set and test set
data_dir = pathlib.Path('data')
test_data_dir = pathlib.Path('test_data')

if not data_dir.exists():
    tf.keras.utils.get_file(
        'speech_commands_v0.02.tar.gz',
        origin="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        extract=True,
        cache_dir='.', cache_subdir='data')

if not test_data_dir.exists():
    tf.keras.utils.get_file(
        'speech_commands_test_set_v0.02.tar.gz',
        origin="http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz",
        extract=True,
        cache_dir='.', cache_subdir='test_data')

# Delete unneccesary files
if os.path.isfile('data/LICENSE'):
    os.remove('data/LICENSE')
if os.path.isfile('data/validation_list.txt'):
    os.remove('data/validation_list.txt')
if os.path.isfile('data/testing_list.txt'):
    os.remove('data/testing_list.txt')
if os.path.isfile('data/.DS_Store'):
    os.remove('data/.DS_Store')
if os.path.isfile('data/speech_commands_v0.02.tar.gz'):
    os.remove('data/speech_commands_v0.02.tar.gz')
if os.path.isdir('data/_background_noise_/'):
    file_list = os.listdir('data/_background_noise_/')
    for i, file_p in enumerate(file_list):
        os.remove('data/_background_noise_/'+file_p)
    os.rmdir('data/_background_noise_/')

# See a statistics of the data set and test set
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands(Train&Valid set):', commands)
test_commands = np.array(tf.io.gfile.listdir(str(test_data_dir)))
test_commands = test_commands[test_commands != 'README.md']
print('Commands(Test set):', test_commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of training/validation examples:', num_samples)
print('Number of examples per label:', 
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

test_filenames = tf.io.gfile.glob(str(test_data_dir) + '/*/*')
test_filenames = tf.random.shuffle(test_filenames)
num_test_samples = len(test_filenames)
print('Number of testing examples:', num_test_samples)
print('Number of examples per label:', 
      len(tf.io.gfile.listdir(str(test_data_dir/test_commands[0]))))
print('Example file tensor:', test_filenames[0])

# Splitting data set
train_files = filenames[:TRAIN_SET_SIZE]
val_files = filenames[TRAIN_SET_SIZE:]
test_files = test_filenames[:]

print()
print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

exit()
# Convert audio_binary into tensor
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph
    return parts[-2]

# Get waveform data and label
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

# Convert waveform into spectogram
def get_spectogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectogram = tf.abs(spectogram)

    return spectogram

# Parallel process of converting audio_binary into waveforms
AUTOTUNE = tf.data.experimental.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

# Plotting sample waveforms
PLOT_WAVEFORM = False
if PLOT_WAVEFORM:
    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(waveform_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)
    
    plt.show()

# Take a sample from waveform dataset and get the input size
for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectogram = get_spectogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectogram shape:', spectogram.shape)
print('Audio playback')
display.display(display.Audio(waveform, rate=16000))

def plot_spectogram(spectogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns). An epsilon is added to avoid log of zero.
    log_spec = np.log(spectogram.T+np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def get_spectogram_and_label_id(audio, label):
    spectogram = get_spectogram(audio)
    spectogram = tf.expand_dims(spectogram, -1)
    label_id = tf.argmax(label == commands)
    return spectogram, label_id

# Plotting Sample Waveform and Spectogram
PLOT_SPECTOGRAM = False
if PLOT_SPECTOGRAM:
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    plot_spectogram(spectogram.numpy(), axes[1])
    axes[1].set_title('Spectogram')
    plt.show()

# Parallel process of converting waveforms into spectograms
spectogram_ds = waveform_ds.map(
    get_spectogram_and_label_id, num_parallel_calls=AUTOTUNE)

# Plotting Sample Spectograms
PLOT_SPECTOGRAM = False
if PLOT_SPECTOGRAM:
    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, (spectogram, label_id) in enumerate(spectogram_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectogram(np.squeeze(spectogram.numpy()), ax)
        ax.set_title(commands[label_id.numpy()])
        ax.axis('off')
    
    plt.show()

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds

train_ds = spectogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectogram, _ in spectogram_ds.take(1):
    input_shape = spectogram.shape
print()
print('Input shape:', input_shape)
num_labels = len(commands)
print('Number of Labels', num_labels)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

if(TRAIN_ENABLED):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
    )
    
    model.save_weights(Model_Saved)

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
else:
    model.load_weights(Model_Saved)

test_audio = []
test_labels = []

for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis = 1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

# Plotting label confusion matrix
PLOT_CONFUSION_MTX = 0
if PLOT_CONFUSION_MTX:
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands
                , annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

# Plotting a prediction result of a sample.
PLOT_PRED_SAMPLE = 0
if PLOT_PRED_SAMPLE:
    sample_file = data_dir/'no/01bb6a2a_nohash_0.wav'
    sample_ds = preprocess_dataset([str(sample_file)])
    
    for spectogram, label in sample_ds.batch(1):
        prediction = model(spectogram)
        plt.bar(commands, tf.nn.softmax(prediction[0]))
        plt.title(f'Predictions for "{commands[label[0]]}"')
        plt.show()

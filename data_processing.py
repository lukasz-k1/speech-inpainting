import numpy as np
import data_utils
import os

data_utils.get_data(source="url", folder_path="VCTK_data/")
DATASET_PATH = 'VCTK_data/VCTK-Corpus/wav48'
data_utils.preprocessing(DATASET_PATH)

speakers_ID_list_train = [225, 227, 228, 229, 230, 231, 232, 233, 236,
237, 239, 241, 244, 246, 247, 249, 250, 251, 252, 253, 255, 256, 258, 
259, 261, 262, 263, 264, 267, 268, 269, 270, 271, 273, 274, 277, 278, 
280, 283, 284, 285, 286, 287, 288, 292, 293, 295, 299, 300, 301, 303, 
304, 305, 306, 308, 311, 312, 313, 314, 315, 318, 323, 326, 329, 330, 
333, 336, 339, 340, 341, 343, 347, 351, 360, 362, 363, 364, 374, 376]
speakers_ID_list_validate = [234, 238, 243, 254, 257, 260, 266, 276, 
281, 297, 298, 302, 307, 310, 345]
speakers_ID_list_test = [226, 240, 245, 248, 265, 272, 275, 279, 282, 
294, 316, 317, 334, 335, 361]

wav_paths = []
filenames = []
for directory, subdirs, files in os.walk(DATASET_PATH):
    wav_paths.extend([os.path.join(directory, file) for file in files if file.endswith('.wav')])
    filenames.extend([file for file in files if file.endswith('.wav')])

train_files = []
test_files = []
val_files = []
for i, name in enumerate(filenames):
    id = int(name[1:4])
    if id in speakers_ID_list_train:
        train_files = np.append(train_files, wav_paths[i])
    if id in speakers_ID_list_test:
        test_files = np.append(test_files, wav_paths[i])
    if id in speakers_ID_list_validate:
        val_files = np.append(val_files, wav_paths[i])

np.random.shuffle(train_files)

X_train = np.zeros((len(train_files)*1000, 32), dtype=np.float32)
y_train = np.zeros((len(train_files)*1000, 32), dtype=np.float32)

current_index = 0
for i, path in enumerate(train_files):
    signal_mfcc, signal_stds, signal_means, corrupted_signal_mfcc, corrupted_signal_stds, corrupted_signal_means = data_utils.extract_features(path)
    X_train[current_index:current_index+corrupted_signal_mfcc.shape[1],:] = corrupted_signal_mfcc.T
    y_train[current_index:current_index+signal_mfcc.shape[1],:] = signal_mfcc.T
    current_index = current_index + signal_mfcc.shape[1]
X_train = X_train[:current_index,:]
y_train = y_train[:current_index,:]


save_path = "arrays/"
np.save(f'{save_path}X_train', X_train, fix_imports=False)
np.save(f'{save_path}y_train', y_train, fix_imports=False)
del(X_train)
del(y_train)

X_test = np.zeros((len(test_files)*1000, 32), dtype=np.float32)
y_test = np.zeros((len(test_files)*1000, 32), dtype=np.float32)

current_index = 0
for i, path in enumerate(test_files):
    signal_mfcc, signal_stds, signal_means, corrupted_signal_mfcc, corrupted_signal_stds, corrupted_signal_means = data_utils.extract_features(path)
    X_test[current_index:current_index+corrupted_signal_mfcc.shape[1],:] = corrupted_signal_mfcc.T
    y_test[current_index:current_index+signal_mfcc.shape[1],:] = signal_mfcc.T
    current_index = current_index + signal_mfcc.shape[1]
X_test = X_test[:current_index,:]
y_test = y_test[:current_index,:]

np.save(f'{save_path}X_test', X_test, fix_imports=False)
np.save(f'{save_path}y_test', y_test, fix_imports=False)
del(X_test)
del(y_test)

X_val = np.zeros((len(val_files)*1000, 32), dtype=np.float32)
y_val = np.zeros((len(val_files)*1000, 32), dtype=np.float32)
X_val_stds = np.zeros((len(val_files), 32))
X_val_means = np.zeros((len(val_files), 32))
y_val_stds = np.zeros((len(val_files), 32))
y_val_means = np.zeros((len(val_files), 32))
X_val_framecounts = np.zeros(len(val_files))
y_val_framecounts = np.zeros(len(val_files))

current_index = 0
for i, path in enumerate(val_files):
    signal_mfcc, signal_stds, signal_means, corrupted_signal_mfcc, corrupted_signal_stds, corrupted_signal_means = data_utils.extract_features(path)
    X_val_stds[i,:] = corrupted_signal_stds
    X_val_means[i,:] = corrupted_signal_means
    X_val_framecounts[i] = corrupted_signal_mfcc.shape[1]
    y_val_stds[i,:] = signal_stds
    y_val_means[i,:] = signal_means
    y_val_framecounts[i] = signal_mfcc.shape[1]
    X_val[current_index:current_index+corrupted_signal_mfcc.shape[1],:] = corrupted_signal_mfcc.T
    y_val[current_index:current_index+signal_mfcc.shape[1],:] = signal_mfcc.T
    current_index = current_index + signal_mfcc.shape[1]

X_val = X_val[:current_index,:]
y_val = y_val[:current_index,:]

np.save(f'{save_path}X_val', X_val, fix_imports=False)
np.save(f'{save_path}y_val', y_val, fix_imports=False)
np.save(f'{save_path}y_val_stds', y_val_stds, fix_imports=False)
np.save(f'{save_path}y_val_means', y_val_means, fix_imports=False)
np.save(f'{save_path}X_val_stds', X_val_stds, fix_imports=False)
np.save(f'{save_path}X_val_means', X_val_means, fix_imports=False)
np.save(f'{save_path}X_val_framecounts', X_val_framecounts, fix_imports=False)
np.save(f'{save_path}y_val_framecounts', y_val_framecounts, fix_imports=False)
del(X_val)
del(y_val)
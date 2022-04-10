import numpy as np
from scipy import signal

def uniform_corruption_mask(n_frames, freq_bins, corruption_percentage, min_duration, 
                            max_duration, min_freq_width, max_freq_width):

    corruption_mask = np.ones((freq_bins, n_frames))

    mean_num_frames = min_duration + (max_duration - min_duration) / 2
    mean_num_freq_bins = min_freq_width + (max_freq_width - min_freq_width) / 2
    num_gaps = int(np.round(corruption_percentage*freq_bins*n_frames / (mean_num_frames*mean_num_freq_bins)))

    for i in range(num_gaps):
        gap_mid_frame = int(np.round(np.random.randint(0, n_frames + max_duration) - (max_duration/2)))
        gap_mid_freq_bin = int(np.round(np.random.randint(0, freq_bins + max_freq_width) - (max_freq_width/2)))

        gap_duration = np.random.randint(min_duration, max_duration + 1)
        gap_freq_width = np.random.randint(min_freq_width, max_freq_width + 1)

        gap_frames = gap_mid_frame + np.arange(gap_duration) - int(np.round(gap_duration/2))
        gap_freq_bins = gap_mid_freq_bin + np.arange(gap_freq_width) - int(np.round(gap_freq_width/2))
        
        gap_frames = gap_frames[gap_frames >= 0]
        gap_frames = gap_frames[gap_frames < n_frames]
        gap_freq_bins = gap_freq_bins[gap_freq_bins >= 0]
        gap_freq_bins = gap_freq_bins[gap_freq_bins < freq_bins]

        if gap_freq_bins.size == 0 or gap_frames.size == 0:
            continue
        
        rows = gap_freq_bins
        cols = gap_frames

        corruption_mask[np.min(rows):np.max(rows)+1,np.min(cols):np.max(cols)+1] = 0

    return corruption_mask

def corrupt_signal(uncorrupted_signal, corruption_percentage=0.3, min_duration=5, 
                   max_duration=15, min_freq_width=40, max_freq_width=200):

    f, t, uncorrupted_stft = signal.stft(uncorrupted_signal, fs = 16000, nfft=640, noverlap=320, nperseg=640)

    fft_length, max_frame = uncorrupted_stft.shape
    corruption_mask = uniform_corruption_mask(max_frame, fft_length, corruption_percentage, min_duration, max_duration, min_freq_width, max_freq_width)
    corrupted_stft = corruption_mask * uncorrupted_stft

    t, corrupted_signal = signal.istft(corrupted_stft, fs = 16000, nfft=640, noverlap=320, nperseg=640)
    
    return corrupted_signal
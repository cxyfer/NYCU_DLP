import os
import numpy as np
import torch
from scipy import signal


class MIBCI2aDataset(torch.utils.data.Dataset):

    def _getFeatures(self, filePath):
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        X = []
        for filename in os.listdir(filePath):
            _, ext = os.path.splitext(filename)
            if ext == '.npy':
                filepath = os.path.join(filePath, filename)
                data = np.load(filepath)
                X.append(data)
        return np.concatenate(X, axis=0)

    def _getLabels(self, filePath):
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        Y = []
        for filename in os.listdir(filePath):
            _, ext = os.path.splitext(filename)
            if ext == '.npy':
                filepath = os.path.join(filePath, filename)
                data = np.load(filepath)
                Y.append(data)
        return np.concatenate(Y, axis=0)

    def __init__(self, mode, transform=None):
        # remember to change the file path according to different experiments
        mode = mode.lower()
        assert mode in [
            'sd_train', 'loso_train', 'sd_test', 'loso_test', 'finetune'
        ]
        if mode == 'sd_train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            self.features = self._getFeatures(filePath='./dataset/SD_train/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_train/labels/')
        elif mode == 'loso_train':
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
        elif mode == 'sd_test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')
        elif mode == 'loso_test':
            # leave-one-subject-out: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')
        elif mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')

        self.transform = EEGAugmentation() if mode.endswith('train') or mode == 'finetune' else None
        # self.transform = None

    def __len__(self):
        # implement the len method
        return self.features.shape[0]

    def __getitem__(self, idx):
        # implement the getitem method
        # return self.features[idx], self.labels[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features, labels = self.features[idx], self.labels[idx]
        if self.transform is not None:
            features = self.transform.augment(features)
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return features, labels

"""
    EEG Data Augmentation
    Reference: https://arxiv.org/abs/2206.14483
"""
class EEGAugmentation:
    def __init__(self, fs=125): # 採樣頻率為125Hz
        self.fs = fs

    def augment(self, signal):
        augmentation_type = np.random.choice([
            'without_augmentation',
            'gaussian_noise', # 添加高斯噪聲 Add Gaussian noise
            'time_shift', # 時間偏移 Time shift
            # 'amplitude_scale', # 振幅縮放 Amplitude scale -> 感覺是負面效果
            # 'channel_shuffle', # 通道置換 Channel shuffle -> 可能不能用在 EEG 上
            # 'bandpass_filter', # 頻帶濾波 -> TA已經處理過了
            # 'band_suppress', # 隨機頻帶抑制
            # 'ft_surrogate', # 傅立葉變換替代 -> 超慢
        ], p = [0.1, 0.45, 0.45])

        # augmentation_type = 'crop_and_padding' # for test
        # print(augmentation_type)

        if augmentation_type == 'gaussian_noise':
            return self.gaussian_noise(signal)
        if augmentation_type == 'time_shift':
            return self.time_shift(signal)
        if augmentation_type == 'amplitude_scale':
            return self.amplitude_scale(signal)
        if augmentation_type == 'channel_shuffle':
            return self.channel_shuffle(signal)
        if augmentation_type == 'band_suppress':
            return self.band_suppress(signal)
        if augmentation_type == 'bandpass_filter':
            return self.bandpass_filter(signal, self.fs)
        if augmentation_type == 'ft_surrogate':
            return self.ft_surrogate(signal)
        return signal

    @staticmethod
    def gaussian_noise(eeg, std=0.1):
        """
        Add Gaussian noise to the input signal.

        Args:
            eeg (numpy.ndarray): Input signal, shape (n_channels, n_timepoints) or (n_batch, n_channels, n_timepoints)
            std (float): Standard deviation of the Gaussian noise
        Returns:
            numpy.ndarray: Noised signal with the same shape as input
        """
        noise = np.random.normal(0, std, eeg.shape) # (mean, std, size)
        return eeg + noise

    @staticmethod
    def time_shift(eeg, shift_max=10):
        """
        Shift the input signal in time.

        Args:
            eeg (numpy.ndarray): Input signal, shape (n_channels, n_timepoints) or (n_batch, n_channels, n_timepoints)
            shift_max (int): Maximum shift in time
        Returns:
            numpy.ndarray: Shifted signal with the same shape as input
        """
        shift = np.random.randint(-shift_max, shift_max+1) # [-10, 10]
        return np.roll(eeg, shift, axis=-1) # roll at time axis

    @staticmethod
    def amplitude_scale(eeg, scale_range=(0.8, 1.2)):
        """
        Scale the amplitude of the input signal.

        Args:
            eeg (numpy.ndarray): Input signal, shape (n_channels, n_timepoints) or (n_batch, n_channels, n_timepoints)
            scale_range (tuple): Range of the scale factor
        Returns:
            numpy.ndarray: Scaled signal with the same shape as input
        """
        scale = np.random.uniform(*scale_range)
        return eeg * scale

    @staticmethod
    def channel_shuffle(eeg, n_swaps=2):
        """
        Shuffle channels in the EEG signal.
        
        Args:
            eeg (numpy.ndarray): Input EEG signal, shape (n_channels, n_timepoints) or (n_batch, n_channels, n_timepoints)
            n_swaps (int): Number of channel swaps to perform
        Returns:
            numpy.ndarray: EEG signal with shuffled channels, same shape as input
        """
        # 確保輸入是複製的，以避免修改原始數據
        eeg = np.copy(eeg)

        # 檢查輸入維度
        if eeg.ndim == 2:
            n_channels, n_timepoints = eeg.shape
            n_batch = 1
            eeg = eeg.reshape(1, n_channels, n_timepoints)
        elif eeg.ndim == 3:
            n_batch, n_channels, n_timepoints = eeg.shape
        else:
            raise ValueError("Input EEG must be 2D or 3D array")

        for b in range(n_batch):
            for _ in range(n_swaps):
                # 隨機選擇兩個不同的通道
                ch1, ch2 = np.random.choice(n_channels, 2, replace=False)
                # 交換這兩個通道
                eeg[b, ch1], eeg[b, ch2] = eeg[b, ch2].copy(), eeg[b, ch1].copy()

        # 如果原始輸入是2D，則壓縮輸出
        if eeg.shape[0] == 1:
            eeg = eeg.squeeze(0)

        return eeg

    @staticmethod
    def bandpass_filter(eeg, fs, order=5):
        """
        Apply bandpass filter to EEG signal.
        
        Args:
            eeg (numpy.ndarray): Input EEG signal, shape (n_channels, n_timepoints) or (n_batch, n_channels, n_timepoints)
            fs (float): Sampling frequency
            order (int): Order of the filter
        
        Returns:
            numpy.ndarray: Filtered EEG signal, same shape as input
        """
        # 確保輸入是複製的，以避免修改原始數據
        eeg = np.copy(eeg)

        # 檢查輸入維度
        if eeg.ndim == 2:
            n_channels, n_timepoints = eeg.shape
            n_batch = 1
            eeg = eeg.reshape(1, n_channels, n_timepoints)
        elif eeg.ndim == 3:
            n_batch, n_channels, n_timepoints = eeg.shape
        else:
            raise ValueError("Input EEG must be 2D or 3D array")

        # 設計濾波器
        lowcut = np.random.uniform(0.5, 4)  # 隨機選擇低頻截止
        highcut = np.random.uniform(30, 50)  # 隨機選擇高頻截止
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')

        # 應用濾波器
        for i in range(n_batch):
            for j in range(n_channels):
                eeg[i, j] = signal.filtfilt(b, a, eeg[i, j])

        # 如果原始輸入是2D，則壓縮輸出
        if n_batch == 1:
            eeg = eeg.squeeze(0)

        return eeg

    # TO-DO
    @staticmethod
    def band_suppress(eeg, suppress_ratio=0.5):
        return eeg
        f, t, Zxx = eeg.stft(eeg, self.fs)
        mask = np.random.rand(*Zxx.shape[1:]) > suppress_ratio
        Zxx[:, mask] = 0
        _, eeg_suppressed = eeg.istft(Zxx, self.fs)
        return eeg_suppressed

    @staticmethod
    def ft_surrogate(eeg): # Fourier Transform surrogate
        """
        Generate a Fourier Transform surrogate of the input signal.
        
        Args:
            eeg (numpy.ndarray): Input signal, shape (n_channels, n_timepoints) or (n_batch, n_channels, n_timepoints)
        
        Returns:
            numpy.ndarray: Surrogate signal with the same shape as input
        """
        # 確保輸入是複製的，以避免修改原始數據
        eeg = np.copy(eeg)

        # 檢查輸入維度
        if eeg.ndim == 2:
            n_channels, n_timepoints = eeg.shape
            n_batch = 1
        elif eeg.ndim == 3:
            n_batch, n_channels, n_timepoints = eeg.shape
        else:
            raise ValueError("Input signal must be 2D or 3D array")

        # 重塑信號以統一處理
        eeg = eeg.reshape(-1, n_channels, n_timepoints)

        for b in range(n_batch):
            for ch in range(n_channels):
                # 進行傅立葉變換
                fft = np.fft.fft(eeg[b, ch])

                # 提取幅度和相位
                magnitudes = np.abs(fft)

                # 隨機化相位
                random_phases = np.random.uniform(0, 2*np.pi, len(fft))

                # 保持第一個（直流）分量的相位不變
                random_phases[0] = 0

                # 確保共軛對稱性（對於實值信號）
                random_phases[-len(fft)//2+1:] = -random_phases[1:len(fft)//2][::-1]

                # 重建信號
                new_fft = magnitudes * np.exp(1j * random_phases)

                # 逆傅立葉變換
                eeg[b, ch] = np.real(np.fft.ifft(new_fft))

        # 如果原始輸入是2D，則壓縮輸出
        if eeg.shape[0] == 1:
            eeg = eeg.squeeze(0)

        return eeg
    
    @staticmethod
    def crop_and_padding(eeg, crop_size=(400, 438)):
        """
        Crop and padding the input signal.

        Args:
            eeg (numpy.ndarray): Input signal, shape (n_channels, n_timepoints) or (n_batch, n_channels, n_timepoints)
        Returns:
            numpy.ndarray: Cropped and padded signal with the same shape as input
        """

        eeg = np.copy(eeg)
        if eeg.ndim == 2:
            n_channels, n_timepoints = eeg.shape
            n_batch = 1
            eeg = eeg.reshape(1, n_channels, n_timepoints)
        elif eeg.ndim == 3:
            n_batch, n_channels, n_timepoints = eeg.shape
        else:
            raise ValueError("Input signal must be 2D or 3D array")
        
        new_size = np.random.randint(*crop_size)

        # Randomly choose the start position for cropping
        st = np.random.randint(0, eeg.shape[-1] - new_size)
        
        # Perform the cropping
        cropped = eeg[..., st:st + new_size]
        
        # Calculate the padding size
        pad_size = eeg.shape[-1] - new_size

        # Pad the cropped signal
        padded_signal = np.pad(cropped, ((0, 0), (0, 0), (0, pad_size)), mode='constant')

        # Check if the input signal is 2D or 3D
        if n_batch == 1:
            padded_signal = padded_signal.squeeze(0)
        
        return padded_signal

if __name__ == '__main__':
    for mode in ['sd_train', 'loso_train', 'sd_test', 'loso_test', 'finetune']:
        dataset = MIBCI2aDataset(mode=mode)
        print(f'{mode}: {len(dataset)}')
        features, labels = dataset[16:18]
        print(features.shape, labels.shape, labels)
        features, labels = dataset[16]
        print(features.shape, labels.shape, labels)
        x = features.cpu().detach().numpy()

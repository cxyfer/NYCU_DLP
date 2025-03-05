import torch
import numpy as np
import os

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # 讀取所有預處理好的特徵數據，並將它們合併成一個單一的 numpy 陣列
        all_features = []
        for file in os.listdir(filePath):
            if file.endswith('.npy'):
                features = np.load(os.path.join(filePath, file))
                all_features.append(features)
        return np.concatenate(all_features, axis=0)

    def _getLabels(self, filePath):
        # 讀取所有預處理好的標籤數據，並將它們合併成一個單一的 numpy 陣列
        all_labels = []
        for file in os.listdir(filePath):
            if file.endswith('.npy'):
                labels = np.load(os.path.join(filePath, file))
                all_labels.append(labels)
        return np.concatenate(all_labels, axis=0)

    def __init__(self, mode):
        # 根據不同的模式設置文件路徑
        assert mode in ['LOSO_train', 'LOSO_test', 'finetune', 'SD_train', 'SD_test']
        self.mode = mode
        if mode == 'LOSO_train':
            self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
        elif mode == 'finetune':
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')
        elif mode == 'LOSO_test':
            self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')
        elif mode == 'SD_train':
            self.features = self._getFeatures(filePath='./dataset/SD_train/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_train/labels/')
        elif mode == 'SD_test':
            self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')

        # 確保 features 和 labels 的數量是一致的
        assert len(self.features) == len(self.labels)

    def __len__(self):
        # 返回數據集的樣本數量
        return len(self.features)

    def __getitem__(self, idx):
        # 根據索引返回特徵和標籤
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = self.features[idx]
        label = self.labels[idx]
        
        # 如果處於訓練模式，進行數據增強
        if self.mode == 'LOSO_train' or self.mode == 'finetune' or self.mode == 'SD_train':
            # print('augemntation!')
            feature = self.augment(feature)
        
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def augment(self, feature):
        # 添加隨機噪聲
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.1, feature.shape)
            feature = feature + noise

        # 隨機平移
        if np.random.rand() > 0.5:
            shift = np.random.randint(-10, 10)
            feature = np.roll(feature, shift, axis=-1)

        # 隨機裁剪並填充
        if np.random.rand() > 0.5:
            crop_size = np.random.randint(400, 438)
            start = np.random.randint(0, feature.shape[-1] - crop_size)
            cropped_feature = feature[:, start:start+crop_size]
            feature = np.pad(cropped_feature, ((0, 0), (0, feature.shape[-1] - crop_size)), 'constant')

        return feature

# 測試用例
if __name__ == '__main__':
    dataset = MIBCI2aDataset(mode='LOSO_train')
    print(f"Number of samples: {len(dataset)}")
    print(f"Feature shape: {dataset[0][0].shape}, Label: {dataset[0][1]}")
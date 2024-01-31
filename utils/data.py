import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pandas as pd

def data_processing(feather_file_path, NUM_FEATURES):
    df = pd.read_feather(feather_file_path)
    df_filtered = df[df['Label'] != 12]
    df_sorted = df_filtered.sort_values(by='Label')

    y_train = df['Label']

    df = df/255

    X_train = df.drop(['Label', 'flow_id'], axis=1)
    X_train = X_train.to_numpy() / 255
  
   # nhom 20packet thanh 1 flow
    X_train = X_train.reshape(-1,20, NUM_FEATURES)

    y_train = y_train.to_numpy()

   # lay nhan cuoi cung
    y_train = y_train.reshape(-1,20)[:,-1] #em chao anhb
    return X_train, y_train

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class ETCDATA(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        feather_file_path = '../CustomerData/Capture_train_256.feather'
        NUM_FEATURE = 256
        X, y = data_processing(feather_file_path, NUM_FEATURE)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_data, self.train_targets = X_train, y_train
        self.test_data, self.test_targets = X_test, y_test
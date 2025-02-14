from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        image_file = gzip.open(image_filename, 'rb')
        label_file = gzip.open(label_filename, 'rb')
        image_file.read(16)
        label_file.read(8)
        image_data = image_file.read()
        label_data = label_file.read()
        image_file.close()
        label_file.close()
        self.X = np.frombuffer(image_data, dtype=np.uint8).reshape(-1, 28*28).astype(np.float32) / 255.0
        self.y = np.frombuffer(label_data, dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.apply_transforms(self.X[index].reshape(28, 28, -1))
        return x.reshape(-1, 28*28), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
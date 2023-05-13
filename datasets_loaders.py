import os
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TaijiDataset(Dataset):
    def __init__(self, root_dir, transform=None, test_subject=None):
        self.root_dir = root_dir
        self.transform = transform
        self.test_subject = test_subject

        self.samples = []

        subject_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for subject_dir in subject_dirs:
            if self.test_subject is not None and subject_dir == self.test_subject:
                continue  # Skip the test subject

            subject_path = os.path.join(self.root_dir, subject_dir)
            pose_dirs = os.listdir(subject_path)

            for pose_dir in pose_dirs:
                pose_path = os.path.join(subject_path, pose_dir)
                pose_files = [f for f in os.listdir(pose_path) if f.endswith('.jpg')]

                for pose_file in pose_files:
                    img_path = os.path.join(pose_path, pose_file)
                    h5_path = os.path.join(pose_path, pose_file.replace('.jpg', '.h5'))

                    self.samples.append((img_path, h5_path, int(pose_dir)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, h5_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        # Load mocap and foot pressure data
        with h5py.File(h5_path, 'r') as f:
            mocap = torch.tensor(np.nan_to_num(f['MOCAP'][:].flatten()))
            foot_pressure = torch.tensor(np.nan_to_num(f['PRESSURE'][:].flatten()))

        return image, mocap, foot_pressure, label
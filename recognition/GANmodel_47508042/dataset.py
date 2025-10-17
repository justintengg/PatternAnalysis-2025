import os
import numpy as np
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HipMRIDataset(Dataset):
    def __init__(self, root_dir, image_size=256, transform=None, max_slices_per_volume=None, recursive=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.max_slices = max_slices_per_volume
        self.recursive = recursive

        # finding files
        self.files = self.collect_files()
        if len(self.files) == 0:
            raise ValueError(f"no files found in {root_dir}")

        # transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # pre-index slices
        self.index_map = []
        for f_idx, path in enumerate(self.files):
            vol = nib.load(path).get_fdata()
            depth = vol.shape[2]
            n_slices = depth if self.max_slices is None else min(depth, self.max_slices)
            for si in range(n_slices):
                self.index_map.append((f_idx, si))

    def collect_files(self):
        exists = ('.nii', '.nii.gz')
        files = []
        if self.recursive:
            for root, _, filenames in os.walk(self.root_dir):
                for fn in filenames:
                    if fn.lower().endswith(exists):
                        files.append(os.path.join(root, fn))
        else:
            for fn in os.listdir(self.root_dir):
                if fn.lower().endswith(exists):
                    files.append(os.path.join(self.root_dir, fn))
        return sorted(files)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        file_index, slice_index = self.index_map[index]
        path = self.files[file_index]
        vol = nib.load(path).get_fdata()
        min, max = vol.min(), vol.max()
        if max - min < 1e-8:
            norm = np.zeros_like(vol, dtype=np.float32)
        else:
            norm = (vol - min) / (max - min)

        slice2d = norm[:, :, slice_index]
        # convert to uint8 image then to PIL so transform works reliably
        img = Image.fromarray((slice2d * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)  # tensor in [0,1}, shape (C,H,W)
        return img


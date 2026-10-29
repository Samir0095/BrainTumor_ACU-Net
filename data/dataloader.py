import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


RANDOM_SEED = 42
TRAIN_RATIO = 0.80


class BraTSDataset(Dataset):
    """
    BraTS 2018 2D slice dataset.

    Each sample contains:
        Image: (4, H, W) - FLAIR, T1, T1ce, T2
        Mask:  (H, W)    - binary tumor mask
    """

    def __init__(self, patient_list):
        self.samples = []
        self.patient_cache = {}

        for patient_path in patient_list:
            patient_path = Path(patient_path)
            patient_id = patient_path.name

            flair_path = patient_path / f"{patient_id}_flair.nii"
            flair_img = nib.load(flair_path)

            num_slices = flair_img.shape[2]

            for slice_idx in range(num_slices):
                self.samples.append(
                    (patient_path, patient_id, slice_idx)
                )

        print(f"Created dataset with {len(self.samples)} slices.")

    def __len__(self):
        return len(self.samples)

    def _load_patient(self, patient_path, patient_id):
        patient_key = str(patient_path)

        if patient_key in self.patient_cache:
            return self.patient_cache[patient_key]

        modalities = {
            name: nib.load(patient_path / f"{patient_id}_{suffix}.nii")
            for name, suffix in [
                ("flair", "flair"),
                ("t1", "t1"),
                ("t1ce", "t1ce"),
                ("t2", "t2"),
                ("seg", "seg"),
            ]
        }

        self.patient_cache[patient_key] = modalities
        return modalities

    @staticmethod
    def _normalize(modality):
        modality = modality.astype(np.float32, copy=True)
        nonzero = modality != 0

        if np.any(nonzero):
            mean = modality[nonzero].mean()
            std = modality[nonzero].std()

            if std > 0:
                modality[nonzero] = (
                    modality[nonzero] - mean
                ) / std
            else:
                modality[nonzero] = modality[nonzero] - mean

        modality[~nonzero] = 0
        return modality

    def __getitem__(self, index):
        patient_path, patient_id, slice_idx = self.samples[index]

        volumes = self._load_patient(patient_path, patient_id)

        modality_slices = []

        for name in ("flair", "t1", "t1ce", "t2"):
            slice_data = np.asarray(
                volumes[name].dataobj[:, :, slice_idx],
                dtype=np.float32
            )
            modality_slices.append(self._normalize(slice_data))

        image = np.stack(modality_slices, axis=0)

        seg_slice = np.asarray(
            volumes["seg"].dataobj[:, :, slice_idx],
            dtype=np.float32
        )

        # Binary target: background = 0, any tumor label = 1
        mask = (seg_slice > 0).astype(np.float32)

        image = torch.from_numpy(image.copy())
        mask = torch.from_numpy(mask.copy())

        return image, mask


def get_patient_split(dataset_path):
    dataset_path = Path(dataset_path)

    training_path = (
        dataset_path / "MICCAI_BraTS_2018_Data_Training"
    )

    all_patients = []

    for tumor_type in ("HGG", "LGG"):
        tumor_path = training_path / tumor_type

        for patient_path in sorted(tumor_path.iterdir()):
            if patient_path.is_dir():
                all_patients.append(patient_path)

    # Same ordering and shuffle procedure as split_dataset.py
    random.seed(RANDOM_SEED)
    random.shuffle(all_patients)

    split_index = int(len(all_patients) * TRAIN_RATIO)

    train_patients = all_patients[:split_index]
    val_patients = all_patients[split_index:]

    train_ids = {path.name for path in train_patients}
    val_ids = {path.name for path in val_patients}

    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {overlap}")

    return train_patients, val_patients


def get_dataloaders(dataset_path, batch_size=1):
    train_patients, val_patients = get_patient_split(dataset_path)

    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")

    train_dataset = BraTSDataset(train_patients)
    val_dataset = BraTSDataset(val_patients)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


if __name__ == "__main__":
    DATASET_PATH = (
        r"E:\3.2\Brain Tumor Research\BrainTumor_UNet"
        r"\Datasets\BraTS2018"
    )

    print("Testing ACU-Net data loaders")

    train_loader, val_loader = get_dataloaders(
        DATASET_PATH,
        batch_size=1
    )

    print("\nTraining batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    for split_name, loader in (
        ("Training", train_loader),
        ("Validation", val_loader)
    ):
        images, masks = next(iter(loader))

        print(f"\n{split_name} sample batch:")
        print("Images:", tuple(images.shape), images.dtype)
        print("Masks:", tuple(masks.shape), masks.dtype)
        print("Mask values:", torch.unique(masks))

        assert images.ndim == 4
        assert images.shape[1] == 4
        assert masks.ndim == 3
        assert images.shape[0] == masks.shape[0]
        assert torch.isfinite(images).all()
        assert set(torch.unique(masks).tolist()).issubset({0.0, 1.0})

    print("\nDataLoader test PASSED!")

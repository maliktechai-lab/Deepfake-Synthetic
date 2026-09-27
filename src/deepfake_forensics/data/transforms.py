"""
Data transforms and augmentation for deepfake forensics.

For EXP-S01, the preprocessing pipeline already produces standardized 224x224 crops.
The transform pipeline should NOT resize again - only apply augmentation/normalization.
"""

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def get_train_transforms(
    target_size: tuple[int, int] = (224, 224),
    hflip_prob: float = 0.5,
    color_jitter: float = 0.0,
    random_erase_prob: float = 0.0,
    normalize: bool = True,
    resize: bool = False,  # S01: preprocessing already produces 224x224
) -> T.Compose:
    """Get training transforms with configurable augmentation.
    
    For EXP-S01, set resize=False since preprocessing already produces 224x224 crops.
    """
    transforms = []
    
    if resize:
        transforms.append(T.Resize(target_size, interpolation=InterpolationMode.BILINEAR))

    if hflip_prob > 0:
        transforms.append(T.RandomHorizontalFlip(p=hflip_prob))

    if color_jitter > 0:
        transforms.append(T.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=min(0.1, color_jitter / 2),
        ))

    transforms.append(T.ToTensor())

    if normalize:
        # ImageNet normalization
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if random_erase_prob > 0:
        transforms.append(T.RandomErasing(p=random_erase_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3)))

    return T.Compose(transforms)


def get_eval_transforms(
    target_size: tuple[int, int] = (224, 224),
    normalize: bool = True,
    resize: bool = False,  # S01: preprocessing already produces 224x224
) -> T.Compose:
    """Get evaluation transforms (no augmentation).
    
    For EXP-S01, set resize=False since preprocessing already produces 224x224 crops.
    """
    transforms = []
    
    if resize:
        transforms.append(T.Resize(target_size, interpolation=InterpolationMode.BILINEAR))
    
    transforms.append(T.ToTensor())

    if normalize:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def get_inference_transforms(
    target_size: tuple[int, int] = (224, 224),
    resize: bool = False,
) -> T.Compose:
    """Get transforms for inference (same as eval but explicit)."""
    return get_eval_transforms(target_size=target_size, resize=resize)


def get_s01_train_transforms(
    hflip_prob: float = 0.5,
    color_jitter: float = 0.1,
    random_erase_prob: float = 0.1,
    normalize: bool = True,
) -> T.Compose:
    """Get EXP-S01 specific training transforms (no resize, standardized 224x224 input)."""
    return get_train_transforms(
        resize=False,
        hflip_prob=hflip_prob,
        color_jitter=color_jitter,
        random_erase_prob=random_erase_prob,
        normalize=normalize,
    )


def get_s01_eval_transforms(
    normalize: bool = True,
) -> T.Compose:
    """Get EXP-S01 specific evaluation transforms (no resize, standardized 224x224 input)."""
    return get_eval_transforms(resize=False, normalize=normalize)

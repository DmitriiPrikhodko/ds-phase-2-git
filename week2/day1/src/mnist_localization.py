import torch
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
import numpy as np


class ResizedMNISTDataset(Dataset):

    def __init__(self, mnist_dataset, new_size=(100, 100)):
        self.mnist_dataset = mnist_dataset
        self.new_size = new_size

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        image = T.ToPILImage()(image)

        original_size = image.size
        left_pad = np.random.randint(0, self.new_size[0] - original_size[0])
        top_pad = np.random.randint(0, self.new_size[1] - original_size[1])
        right_pad = self.new_size[0] - original_size[0] - left_pad
        bottom_pad = self.new_size[1] - original_size[1] - top_pad

        padded_img = ImageOps.expand(
            image, (left_pad, top_pad, right_pad, bottom_pad), fill=0
        )

        digit_area = (
            left_pad,
            top_pad,
            left_pad + original_size[0],
            top_pad + original_size[1],
        )

        return (
            T.ToTensor()(padded_img),
            torch.Tensor([label]),
            torch.Tensor(digit_area) / self.new_size[0],
        )


def get_dataloaders(generator, batch_size) -> tuple[DataLoader, DataLoader]:
    """function generates two dataloaders with mnist for localization task

    Args:
        batch_size (int, optional): Batch size in dataloaders. Defaults to 32.

    Returns:
        tuple[DataLoader, Dataloader]: train and valid dataloaders
    """
    # Загрузка данных MNIST
    augmentation = (
        # T.ColorJitter(),  # Случайно меняет яркость, контраст, насыщенность и оттенок
        T.RandomRotation((0, 180)),  # Случайно поворачивает изображение
        T.RandomHorizontalFlip(
            p=0.5
        ),  # С вероятностью p отажает изображение по горизонтальной оси
    )
    transform = T.Compose(
        [
            # *augmentation,
            T.ToTensor(),
            # T.Normalize((0.5,), (0.5,)),
        ]
    )
    mnist_train = datasets.MNIST(
        root="data/mnist_detection", train=True, download=True, transform=transform
    )
    mnist_val = datasets.MNIST(
        root="data/mnist_detection", train=False, download=True, transform=transform
    )

    # Создание новых датасетов с увеличенными изображениями
    resized_mnist_train = ResizedMNISTDataset(mnist_train)
    resized_mnist_val = ResizedMNISTDataset(mnist_val)

    # Загрузка данных с использованием DataLoader
    train_loader = DataLoader(
        resized_mnist_train,
        batch_size=batch_size,
        generator=generator,
        shuffle=True,
        num_workers=4,  # подстрой под свои CPU-ядра
        pin_memory=True,  # ускоряет передачу CPU→GPU
        persistent_workers=True,  # не пересоздаёт воркеров на каждой эпохе
        prefetch_factor=2,  # (по умолчанию 2) батчей на воркер вперёд
        drop_last=True,  # убираем неполный батч (оптимальнее для тренировки)
    )
    val_loader = DataLoader(
        resized_mnist_val,
        batch_size=batch_size,
        generator=generator,
        shuffle=False,
        num_workers=4,  # подстрой под свои CPU-ядра
        pin_memory=True,  # ускоряет передачу CPU→GPU
        persistent_workers=True,  # не пересоздаёт воркеров на каждой эпохе
        prefetch_factor=2,  # (по умолчанию 2) батчей на воркер вперёд
    )
    return train_loader, val_loader

import cProfile
import pstats
import torch
import stable_pretraining as spt
from stable_pretraining.data import transforms

train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop(
        224,
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToImage(),
)

train_dataset = spt.data.HFDataset(
    path="uoft-cs/cifar10",
    split="train",
    transform=train_transform,
    rename_columns={"img": "image"},
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, shuffle=True, drop_last=True, batch_size=64
)
profiler = cProfile.Profile()
profiler.enable()
# Run a few batches
for i, batch in enumerate(train_loader):
    if i >= 10:
        break
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)

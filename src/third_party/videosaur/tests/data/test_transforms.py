import torch

from src.third_party.videosaur.videosaur.data import transforms_video


def test_from_tensor():
    size = (10, 100, 100, 3)
    video = torch.randint(0, 255, size, dtype=torch.uint8)
    id_video = transforms_video.from_tensor(transforms_video.to_tensor(video))
    torch.equal(id_video, video)

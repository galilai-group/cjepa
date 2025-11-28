from videosaur.modules import timm
from videosaur.modules.decoders import build as build_decoder
from videosaur.modules.encoders import build as build_encoder
from videosaur.modules.groupers import build as build_grouper
from videosaur.modules.initializers import build as build_initializer
from videosaur.modules.networks import build as build_network
from videosaur.modules.utils import Resizer, SoftToHardMask
from videosaur.modules.utils import build as build_utils
from videosaur.modules.utils import build_module, build_torch_function, build_torch_module
from videosaur.modules.video import LatentProcessor, MapOverTime, ScanOverTime
from videosaur.modules.video import build as build_video

__all__ = [
    "build_decoder",
    "build_encoder",
    "build_grouper",
    "build_initializer",
    "build_network",
    "build_utils",
    "build_module",
    "build_torch_module",
    "build_torch_function",
    "timm",
    "MapOverTime",
    "ScanOverTime",
    "LatentProcessor",
    "Resizer",
    "SoftToHardMask",
]


BUILD_FNS_BY_MODULE_GROUP = {
    "decoders": build_decoder,
    "encoders": build_encoder,
    "groupers": build_grouper,
    "initializers": build_initializer,
    "networks": build_network,
    "utils": build_utils,
    "video": build_video,
    "torch": build_torch_function,
    "torch.nn": build_torch_module,
    "nn": build_torch_module,
}

import numpy as np
import pytest
import timm
import torch

import videosaur.modules  # noqa: F401


@pytest.mark.parametrize(
    "name,inp_shape,outp_shape",
    [
        ("resnet18_savi", (1, 3, 128, 128), (1, 512, 16, 16)),
        ("resnet18_savi", (1, 3, 224, 224), (1, 512, 28, 28)),
        ("resnet34_savi", (1, 3, 128, 128), (1, 512, 16, 16)),
        ("resnet50_savi", (1, 3, 128, 128), (1, 2048, 16, 16)),
    ],
)
def test_savi_resnet(name, inp_shape, outp_shape):
    model = timm.create_model(name, pretrained=False)

    found_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert not found_bn

    with torch.no_grad():
        inp = torch.randn(*inp_shape)
        outp = model.forward_features(inp)

    assert outp.shape == outp_shape


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,inp_shape,outp_shape",
    [
        ("resnet50_dino", (1, 3, 224, 224), (1, 2048, 7, 7)),
        ("resnet50_mocov3", (1, 3, 224, 224), (1, 2048, 7, 7)),
        ("vit_base_patch4_224_msn", (1, 3, 224, 224), (1, 3137, 768)),
    ],
)
def test_custom_pretrained_models(name, inp_shape, outp_shape):
    """Check that custom models with pre-trained weights load properly."""
    model = timm.create_model(name, pretrained=True)

    with torch.no_grad():
        inp = torch.randn(*inp_shape)
        outp = model.forward_features(inp)

    assert outp.shape == outp_shape


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,inp_shape,outp_shape",
    [
        ("vit_base_patch16_224_dino_timetuning", (1, 3, 224, 224), (1, 197, 768)),
        ("vit_base_patch8_224_dino_timetuning", (1, 3, 224, 224), (1, 785, 768)),
    ],
)
def test_vit_dino_timetuning(name, inp_shape, outp_shape):
    model = timm.create_model(name, pretrained=True)

    found_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert not found_bn

    with torch.no_grad():
        inp = torch.randn(*inp_shape)
        outp = model.forward_features(inp)

    assert outp.shape == outp_shape


# Expected outputs of custom ViT weights. We store them verbatim here to avoid putting binary
# files into git. We store them as strings instead of numbers to avoid black formatting them.
CUSTOM_VIT_WEIGHTS_RESULTS = {
    "vit_base_patch16_224_mae/zeros/0_0-20_0": """
[0.28719592, 0.24583483, 0.26907375, 0.30731523, 0.2868713 , 0.32485905, 0.30731025,
0.33331153, 0.34344897, 0.33817446, 0.3270159 , 0.29424447, 0.27153555, 0.3556132 ,
0.19864245, 0.1820898 , 0.2051502 , 0.2437797 , 0.20441592, 0.24913353]""",
    "vit_base_patch16_224_mae/zeros/0_0_0-20": """
[0.28719592, -0.0576665 , -0.72432476, -0.18103008,  0.08990244, -0.9472992 ,  0.6524778 ,
0.17858694, -0.08992696,  0.06529036,  0.05850611, -0.30322337, -0.09932699,  2.2201734 ,
-0.02982868, -0.09614885, -0.15114741, -0.61356527,  0.58381927,  0.08191889]""",
    "vit_base_patch16_224_mae/ones/0_0-20_0": """
[0.2549035 , 0.28759107, 0.31177142, 0.32582435, 0.30852506, 0.34220454, 0.32828912,
0.35779768, 0.3720604 , 0.34713408, 0.32903695, 0.29814035, 0.28354186, 0.33921587,
0.15334876, 0.21293344, 0.23456964, 0.2595923 , 0.22746845, 0.2685317]""",
    "vit_base_patch16_224_mae/ones/0_0_0-20": """
[0.2549035 , -0.09336171, -0.6880767 , -0.28376287,  0.02024214, -0.76264304,  0.5904238 ,
0.15652883, -0.12118589,  0.21486482,  0.15963039, -0.27727827,  0.03670263,  1.8414731 ,
0.07228839, -0.20896748, -0.27817705, -0.485394  ,  0.6534163 , -0.18775147]""",
    "vit_base_patch16_224_mocov3/zeros/0_0-20_0": """
[-0.05149402, -0.03026696, -0.02507262, -0.02068919, -0.01894361, -0.02121132, -0.02440684,
-0.0279355 , -0.02931937, -0.02867937, -0.02717797, -0.02422431, -0.02747217, -0.05334421,
-0.02532654, -0.0125381 , -0.00969828, -0.00759022, -0.00619568, -0.00793152]""",
    "vit_base_patch16_224_mocov3/zeros/0_0_0-20": """
[-0.05149402,  0.01578675, -0.09754238,  0.00310562,  0.04205038,  0.00287579,  0.02752389,
-0.08285441, -0.02746764,  0.00113256, -0.02433307,  0.03415485, -0.0008044 ,  0.03292493,
0.06440924,  0.04894995, -0.20814528, -0.08244848, -0.0776324 , -0.0591822]""",
    "vit_base_patch16_224_mocov3/ones/0_0-20_0": """
[-0.05119515, -0.03057244, -0.02808539, -0.02491217, -0.02365276, -0.024435  , -0.02549231,
-0.02861489, -0.03086424, -0.03028261, -0.02863837, -0.02775326, -0.03039749, -0.05470654,
-0.02382995, -0.00736903, -0.00620988, -0.00577518, -0.00560756, -0.00666199]""",
    "vit_base_patch16_224_mocov3/ones/0_0_0-20": """
[-5.1195148e-02,  2.4619440e-02, -8.7654129e-02, -8.9437347e-03,  4.8605293e-02,
6.5571134e-05,  1.6089279e-02, -8.1912845e-02, -2.9642873e-02,  1.2495698e-02,
-2.6932344e-02,  4.8234582e-02,  1.3394909e-03,  2.9346963e-02,  5.3340483e-02,
5.6126781e-02, -2.1160576e-01, -7.4558653e-02, -8.7223224e-02, -5.3826150e-02]""",
    "vit_base_patch16_224_msn/zeros/0_0-20_0": """
[13.323833, 13.172627, 13.217373, 13.198288, 13.362397, 13.165453, 13.22209 , 13.255227,
13.191107, 13.246759, 13.293469, 13.505752, 13.324869, 13.397531, 13.242817, 13.246699,
13.402661, 13.280356, 13.380144, 13.23673 ]""",
    "vit_base_patch16_224_msn/zeros/0_0_0-20": """
[ 13.323833 , -19.434715 ,  10.907945 ,  -5.5864744, -23.869556 ,  -8.801638 , -14.60782  ,
-9.762392 , -17.333504 ,   8.255657 ,  -8.370134 ,  -7.5889435,   2.4847035,  -4.690354 ,
4.596179 ,  23.206505 , -18.972391 , -10.831766 ,  10.448822 ,  -8.638037]""",
    "vit_base_patch16_224_msn/ones/0_0-20_0": """
[13.377316 , 13.208665 , 13.225736 , 13.205421 , 13.411082 , 13.20577  , 13.237612 ,
13.2523365, 13.221363 , 13.261968 , 13.332789 , 13.483296 , 13.371156 , 13.483837 ,
13.264603 , 13.260244 , 13.37155  , 13.286394 , 13.377354 , 13.243222]""",
    "vit_base_patch16_224_msn/ones/0_0_0-20": """
[ 13.377316 , -19.428873 ,  10.922306 ,  -5.6203094, -23.830498 ,  -8.848695 , -14.560103 ,
-9.800345 , -17.539894 ,   8.266435 ,  -8.417096 ,  -7.53442  ,   2.5665472,  -4.787496 ,
4.625372 ,  23.25421  , -18.970985 , -10.652556 ,  10.562778 ,  -8.680251 ]""",
}


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, inp_type, dims, expected_result",
    [(*key.split("/"), val) for key, val in CUSTOM_VIT_WEIGHTS_RESULTS.items()],
)
def test_custom_pretrained_vits_output(name, inp_type, dims, expected_result):
    """Check that output of custom ViTs still aligns with results from their original repository."""
    model = videosaur.modules.build_encoder(
        {
            "name": "TimmExtractor",
            "model": name,
            "pretrained": True,
            "frozen": True,
            "features": "vit_output",
        }
    )

    if inp_type == "zeros":
        inp = torch.zeros(1, 3, 224, 224)
    elif inp_type == "ones":
        inp = torch.ones(1, 3, 224, 224)

    with torch.no_grad():
        output = model(inp)

    expected = np.array(eval(expected_result), dtype=np.float32)

    slices = []
    for dim in dims.split("_"):
        if dim == "all":
            slices.append(slice(None))
        elif "-" in dim:
            dim = dim.split("-")
            slices.append(slice(int(dim[0]), int(dim[1])))
        else:
            slices.append(slice(int(dim), int(dim) + 1))

    assert np.allclose(output[slices].squeeze().numpy(), expected, atol=1e-5)

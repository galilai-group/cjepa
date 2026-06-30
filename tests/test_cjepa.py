import torch

from cjepa import CJEPA, MaskedSlotPredictor


def test_predictor_shapes_and_object_masking():
    model = MaskedSlotPredictor(
        num_slots=6,
        slot_dim=16,
        history_frames=3,
        pred_frames=2,
        num_masked_slots=2,
        num_unmaskable_slots=2,
        depth=1,
        heads=2,
        mlp_dim=32,
        dropout=0.0,
    )
    inputs = torch.randn(2, 3, 6, 16)
    output, masked = model(inputs)
    assert output.shape == (2, 5, 6, 16)
    assert masked.shape == (2,)
    assert torch.all(masked < 4)
    assert model.inference(inputs).shape == (2, 2, 6, 16)


def test_preextracted_slots_and_planning_cost():
    model = CJEPA(
        num_object_slots=4,
        slot_dim=16,
        history_frames=3,
        pred_frames=1,
        num_masked_slots=1,
        depth=1,
        heads=2,
        mlp_dim=32,
        dropout=0.0,
        action_dim=2,
        action_frameskip=2,
        proprio_dim=4,
    )
    candidates = 3
    info = {
        "slots": torch.randn(1, candidates, 1, 4, 16),
        "goal_slots": torch.randn(1, candidates, 4, 16),
        "proprio": torch.randn(1, candidates, 1, 4),
    }
    actions = torch.randn(1, candidates, 2, 4)
    costs = model.get_cost(info, actions)
    assert costs.shape == (1, candidates)
    assert torch.isfinite(costs).all()


def test_slot_batch_never_requires_object_encoder():
    model = CJEPA(
        num_object_slots=3,
        slot_dim=8,
        history_frames=2,
        pred_frames=1,
        num_masked_slots=1,
        depth=1,
        heads=2,
        mlp_dim=16,
        dropout=0.0,
    )
    slots = torch.randn(2, 3, 3, 8)
    assert torch.equal(model.encode({"slots": slots}), slots)

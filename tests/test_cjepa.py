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
        "goal_proprio": torch.randn(1, candidates, 1, 4),
    }
    actions = torch.randn(1, candidates, 2, 4)
    costs = model.get_cost(info, actions)
    assert costs.shape == (1, candidates)
    assert torch.isfinite(costs).all()


def test_rollout_keeps_original_observation_and_predicts_post_action_state():
    model = CJEPA(
        num_object_slots=2,
        slot_dim=8,
        history_frames=3,
        pred_frames=1,
        num_masked_slots=1,
        depth=1,
        heads=2,
        mlp_dim=16,
        dropout=0.0,
        action_dim=2,
        action_frameskip=1,
        proprio_dim=4,
    ).eval()
    candidates = 3
    info = {
        "slots": torch.randn(1, candidates, 1, 2, 8),
        "goal_slots": torch.randn(1, candidates, 1, 2, 8),
        "proprio": torch.randn(1, candidates, 1, 4),
        "goal_proprio": torch.randn(1, candidates, 1, 4),
    }
    actions = torch.randn(1, candidates, 2, 2)
    rollout = model.rollout(info, actions)

    # One observed frame + one rollout for action 1 + a final state after it.
    assert rollout.shape == (1, candidates, 3, 4, 8)
    expected_action_zero = model.action_encoder(actions.flatten(0, 1)[:, :1])
    actual_action_zero = rollout[:, :, 0, model.action_node].flatten(0, 1)
    assert torch.allclose(actual_action_zero, expected_action_zero[:, 0])


def test_goal_cost_adds_proprio_mse_to_hungarian_object_cost():
    model = CJEPA(
        num_object_slots=2,
        slot_dim=4,
        history_frames=1,
        pred_frames=1,
        num_masked_slots=1,
        depth=1,
        heads=1,
        mlp_dim=8,
        dropout=0.0,
        proprio_dim=2,
    )
    objects = torch.zeros(1, 3, 2, 4)
    goal_objects = torch.zeros_like(objects)
    predicted_proprio = torch.ones(1, 3, 4)
    goal_proprio = torch.zeros_like(predicted_proprio)

    assert torch.equal(
        model.criterion(
            objects,
            goal_objects,
            predicted_proprio,
            goal_proprio,
        ),
        torch.ones(1, 3),
    )


def test_planning_cache_uses_state_identity_not_reused_storage_address():
    model = CJEPA(
        num_object_slots=2,
        slot_dim=4,
        history_frames=1,
        pred_frames=1,
        num_masked_slots=1,
        depth=1,
        heads=1,
        mlp_dim=8,
        dropout=0.0,
    )
    info = {
        "slots": torch.randn(1, 2, 1, 2, 4),
        "goal_slots": torch.randn(1, 2, 1, 2, 4),
        "id": torch.tensor([[[7]], [[7]]]).transpose(0, 1),
        "step_idx": torch.zeros(1, 2, 1, dtype=torch.long),
    }
    model._planning_inputs(info)
    first_key = next(iter(model._planning_cache))

    # The slots tensor and its data_ptr are unchanged, but this is a new state.
    info["step_idx"].add_(25)
    model._planning_inputs(info)
    second_key = next(iter(model._planning_cache))
    assert first_key != second_key


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

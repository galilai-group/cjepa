import torch

from src.third_party.videosaur.videosaur.modules import video


def test_map_over_time():
    class SingleModule(torch.nn.Module):
        def forward(self, inp):
            assert inp.ndim == 2
            return inp

    batch_size, seq_len, dims = 3, 4, 5

    time_mapper = video.MapOverTime(SingleModule())

    inp = torch.ones(batch_size, seq_len, dims)
    assert torch.allclose(time_mapper(inp), inp)


def test_scan_over_time():
    class RecurrentCell(torch.nn.Module):
        def forward(self, state, inputs):
            assert state.ndim == 2
            assert inputs.ndim == 2
            state_next = state + inputs
            return {"state": state, "state_next": state_next, "aux": {"state_next": state_next}}

    batch_size, seq_len, dims = 3, 4, 1
    scanner = video.ScanOverTime(RecurrentCell(), next_state_key="state_next", pass_step=False)

    initial_state = torch.zeros(batch_size, dims)
    inputs = torch.ones(batch_size, seq_len, dims)

    outputs = scanner(initial_state, inputs)

    assert torch.allclose(outputs["state"][:, 0], initial_state)
    assert torch.allclose(outputs["state"][:, 1:], torch.cumsum(inputs, dim=1)[:, :-1])
    assert torch.allclose(outputs["state_next"], torch.cumsum(inputs, dim=1))
    assert torch.allclose(outputs["aux"]["state_next"], torch.cumsum(inputs, dim=1))

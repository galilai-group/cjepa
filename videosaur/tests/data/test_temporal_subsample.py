import numpy as np

from videosaur.videosaur.data.transforms import TemporalSubsample


# def test_temporal_subsample_basic():
#     arr = np.arange(8 * 2 * 2 * 3, dtype=np.uint8).reshape(8, 2, 2, 3)
#     ts = TemporalSubsample(3)
#     out = ts(arr)
#     # should pick frames 0,3,6 -> total 3 frames
#     assert out.shape[0] == 3
#     assert np.array_equal(out[0], arr[0])
#     assert np.array_equal(out[1], arr[3])
#     assert np.array_equal(out[2], arr[6])


# def test_temporal_subsample_stride_one_returns_same():
#     arr = np.random.randint(0, 255, size=(5, 4, 4, 3), dtype=np.uint8)
#     ts = TemporalSubsample(1)
#     out = ts(arr)
#     assert out.shape == arr.shape
#     assert np.array_equal(out, arr)


def test_temporal_subsample_pipeline_helper():
    from videosaur.videosaur.data.pipelines import temporal_subsample_keys

    sample = {"__key__": "0", "video": np.arange(16 * 1 * 1 * 3, dtype=np.uint8).reshape(16, 1, 1, 3)}
    out = temporal_subsample_keys(sample.copy(), keys_to_subsample=("video",), stride=2)
    assert out["video"].shape[0] == 8
    # check first few indices match
    assert np.array_equal(out["video"][0], sample["video"][0])
    assert np.array_equal(out["video"][1], sample["video"][2])

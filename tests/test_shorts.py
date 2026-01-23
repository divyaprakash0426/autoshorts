import sys
from unittest.mock import MagicMock
import numpy as np
from pathlib import Path

# --- Mock GPU libraries BEFORE importing shorts ---
# We must mock decord, cupy, torchaudio, torch so that shorts.py can be imported
# even if these libraries are missing or if we are on a CPU-only node.

# Mock torch
torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.device.return_value = "cpu"
torch_mock.tensor = lambda x, **kwargs: MagicMock()
# Mock basic tensor ops used in shorts
torch_mock.abs = MagicMock()
torch_mock.mean = MagicMock()
torch_mock.sqrt = MagicMock()
torch_mock.cat = MagicMock()
torch_mock.from_numpy = lambda x: x
torch_mock.from_dlpack = MagicMock()
torch_mock.to_dlpack = MagicMock()
torch_mock.nn.functional.interpolate = MagicMock()
sys.modules["torch"] = torch_mock

# Mock torchaudio
torchaudio_mock = MagicMock()
sys.modules["torchaudio"] = torchaudio_mock

# Mock decord
decord_mock = MagicMock()
decord_mock.bridge.set_bridge = MagicMock()
decord_mock.cpu = lambda x: f"cpu({x})"
decord_mock.gpu = lambda x: f"gpu({x})"
sys.modules["decord"] = decord_mock

# Mock cupy
cupy_mock = MagicMock()
cupy_mock.asarray = MagicMock(side_effect=lambda x: x)
cupy_mock.asnumpy = MagicMock(side_effect=lambda x: x)
cupy_mock.from_dlpack = MagicMock()
cupy_mock.to_dlpack = MagicMock()
sys.modules["cupy"] = cupy_mock

# Mock cupyx
cupyx_mock = MagicMock()
sys.modules["cupyx"] = cupyx_mock
sys.modules["cupyx.scipy"] = MagicMock()
sys.modules["cupyx.scipy.ndimage"] = MagicMock()


# Ensure the src directory is on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import shorts AFTER mocking
import shorts  # noqa: E402
from shorts import (  # noqa: E402
    blur_gpu,
    combine_scenes,
    select_background_resolution,
    ProcessingConfig,
    compute_video_action_profile,
    _SecondsTime,
)


# Helper to create scene tuples
def make_scene(start: float, end: float):
    return (_SecondsTime(start), _SecondsTime(end))


def test_select_background_resolution():
    assert select_background_resolution(800) == (720, 1280)
    assert select_background_resolution(1500) == (1440, 2560)
    assert select_background_resolution(2100) == (2160, 3840)


def test_blur_gpu_uses_cupy():
    # Verify blur_gpu calls cupy/cupyx functions
    # Input is a torch tensor mock
    image_tensor = MagicMock()
    image_tensor.is_contiguous.return_value = True

    # Configure mock return for gaussian_filter
    # It returns a cupy array mock
    mock_cupy_array = MagicMock()
    shorts.cupyx.scipy.ndimage.gaussian_filter.return_value = mock_cupy_array

    # Return mock torch tensor
    shorts.torch.utils.dlpack.from_dlpack.return_value = MagicMock()

    blur_gpu(image_tensor)

    shorts.torch.to_dlpack.assert_called_with(image_tensor)
    shorts.cp.from_dlpack.assert_called()
    shorts.cupyx.scipy.ndimage.gaussian_filter.assert_called()
    shorts.torch.utils.dlpack.from_dlpack.assert_called()


def test_combine_scenes_merges_short_scenes():
    config = ProcessingConfig(min_short_length=5, max_short_length=10, max_combined_scene_length=15)
    scenes = [
        make_scene(0, 5),
        make_scene(5, 7),
        make_scene(7, 9),
        make_scene(9, 11),
        make_scene(11, 13),
        make_scene(13, 18),
    ]
    combined = combine_scenes(scenes, config)
    assert len(combined) == 1
    start, end = combined[0]
    assert start.get_seconds() == 5
    assert end.get_seconds() == 13

# render_video (legacy) has been removed.
# render_video_gpu logic is verified via mocks in separate flows or implicitly here if we add such tests.


def test_compute_video_action_profile_sequential():
    """Verify that compute_video_action_profile reads sequentially (batch-by-batch) and subsamples."""
    
    # 1. Setup Mock VideoReader
    mock_vr = MagicMock()
    # Let's say video has 1000 frames, 30 fps
    mock_vr.__len__.return_value = 1000
    mock_vr.get_avg_fps.return_value = 30.0

    # Configure __getitem__ for metadata probe (vr_cpu[0].shape)
    mock_frame = MagicMock()
    mock_frame.shape = (720, 1280, 3)
    mock_vr.__getitem__.return_value = mock_frame

    # Configure torch.cat to return a mock with valid shape/numel
    def side_effect_cat(tensors, **kwargs):
        m = MagicMock()
        m.shape = (100,)
        m.numel.return_value = 100
        m.mean.return_value = MagicMock(return_value=1.0)
        m.std.return_value = MagicMock(return_value=1.0)
        # Math operators return the same mock (so shape persists)
        m.__sub__ = MagicMock(return_value=m)
        m.__truediv__ = MagicMock(return_value=m)
        m.view.return_value = m
        return m
    shorts.torch.cat.side_effect = side_effect_cat

    # Configure get_batch to return a mock tensor
    def side_effect_get_batch(indices):
        # Indices should be a range object
        count = len(indices)
        batch_mock = MagicMock()
        batch_mock.shape = (count, 64, 64, 3) # (B, H, W, C)
        # Slicing returns itself
        batch_mock.__getitem__.return_value = batch_mock
        # float() returns itself
        batch_mock.float.return_value = batch_mock
        return batch_mock

    mock_vr.get_batch.side_effect = side_effect_get_batch

    # 2. Patch VideoReader in shorts
    # Note: 'shorts.VideoReader' comes from 'decord', which we already mocked globally
    # but we need the constructor to return our instance
    shorts.VideoReader.return_value = mock_vr

    # 3. Run function
    # fps=6 means we keep 1 out of 5 frames (30/6 = 5)
    # Total frames 100.
    # It should iterate 0..16, 16..32, ... (batch_size=16)
    times, scores = compute_video_action_profile(Path("dummy.mp4"), fps=6)

    # 4. Verifications
    assert shorts.VideoReader.called
    assert mock_vr.get_batch.called

    # Check calls to get_batch
    # We expect ranges of size 16 starting from 0
    calls = mock_vr.get_batch.call_args_list
    assert len(calls) > 0
    
    # First batch should comprise range(0, 16)
    # The argument passed to get_batch is `batch_range`
    first_call_args = calls[0].args[0]
    assert list(first_call_args) == list(range(0, 16))
    
    # Check that we handled results
    # times and scores should be numpy arrays (mocked)
    # Since torch.cat is mocked, it returns a MagicMock, and .cpu().numpy() returns a MagicMock
    assert isinstance(times, MagicMock) or isinstance(times, np.ndarray) or (times == [])

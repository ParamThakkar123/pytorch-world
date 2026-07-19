import pytest
import numpy as np
from pathlib import Path

h5py = pytest.importorskip("h5py")


class TestTinyWorldsDataset:
    """Tests for TinyWorlds dataset loading and shape handling."""

    def _create_mock_h5(self, tmp_path: Path, layout: str, raw_shape: tuple) -> Path:
        """Create a mock .h5 file for testing."""
        path = tmp_path / f"test_{layout}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "frames", data=np.random.randint(0, 255, raw_shape).astype(np.uint8)
            )
        return path

    def test_output_shape_is_CTHW(self, tmp_path):
        """Test that dataset output is (C, T, H, W) per Genie convention."""
        from world_models.datasets.tinyworlds import TinyWorldsDataset

        h5_path = self._create_mock_h5(tmp_path, "NTHWC", (10, 16, 64, 64, 3))

        class TestLoader(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestLoader.__new__(TestLoader)
        dataset.dataset_name = "TEST"
        dataset.config = TestLoader.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        sample = dataset[0]
        assert sample.shape == (3, 16, 64, 64), (
            f"Expected (C,T,H,W)=(3,16,64,64) but got {sample.shape}"
        )

    def test_grayscale_to_rgb(self, tmp_path):
        """Test that grayscale (1 channel) is converted to RGB (3 channels)."""
        from world_models.datasets.tinyworlds import TinyWorldsDataset

        h5_path = self._create_mock_h5(tmp_path, "NTHW", (5, 16, 64, 64))

        class TestGrayDataset(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestGrayDataset.__new__(TestGrayDataset)
        dataset.dataset_name = "TEST"
        dataset.config = TestGrayDataset.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        sample = dataset[0]
        assert sample.shape[0] == 3, (
            f"Grayscale should become 3 channels, got shape {sample.shape}"
        )

    def test_nthw_layout_handling(self, tmp_path):
        """Test that NTHW layout (no channel dim) is handled correctly."""
        from world_models.datasets.tinyworlds import TinyWorldsDataset

        h5_path = self._create_mock_h5(tmp_path, "NTHW", (5, 16, 64, 64))

        class TestNTHWDataset(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestNTHWDataset.__new__(TestNTHWDataset)
        dataset.dataset_name = "TEST"
        dataset.config = TestNTHWDataset.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        assert dataset.data_layout == "NTHW"
        sample = dataset[0]
        assert sample.shape == (3, 16, 64, 64), (
            f"NTHW should produce (C,T,H,W)=(3,16,64,64) but got {sample.shape}"
        )

    def test_nthwc_layout_handling(self, tmp_path):
        """Test that NTHWC layout is handled correctly."""
        from world_models.datasets.tinyworlds import TinyWorldsDataset

        h5_path = self._create_mock_h5(tmp_path, "NTHWC", (5, 16, 64, 64, 3))

        class TestNTHWCDataset(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestNTHWCDataset.__new__(TestNTHWCDataset)
        dataset.dataset_name = "TEST"
        dataset.config = TestNTHWCDataset.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        assert dataset.data_layout == "NTHWC"
        sample = dataset[0]
        assert sample.shape == (3, 16, 64, 64), (
            f"NTHWC should produce (C,T,H,W)=(3,16,64,64) but got {sample.shape}"
        )

    def test_batch_dimensions_with_dataloader(self, tmp_path):
        """Test that DataLoader produces correct batch dimensions."""
        from world_models.datasets.tinyworlds import TinyWorldsDataset
        from torch.utils.data import DataLoader

        h5_path = self._create_mock_h5(tmp_path, "NTHWC", (20, 16, 64, 64, 3))

        class TestBatchDataset(TinyWorldsDataset):
            DATASET_CONFIGS = {
                "TEST": {
                    "repo_id": "test",
                    "filename": h5_path.name,
                    "description": "test",
                }
            }

        dataset = TestBatchDataset.__new__(TestBatchDataset)
        dataset.dataset_name = "TEST"
        dataset.config = TestBatchDataset.DATASET_CONFIGS["TEST"]
        dataset.num_frames = 16
        dataset.image_size = 64
        dataset.split = "train"
        dataset.cache_dir = str(tmp_path)
        dataset.data_file = None
        dataset._data_file = None
        dataset.num_samples = 0
        dataset.video_length = 0
        dataset._load_data(h5_path)

        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        B, C, T, H, W = batch.shape
        assert B == 4, f"Batch size should be 4, got {B}"
        assert C == 3, f"Channels should be 3 (RGB), got {C}"
        assert T == 16, f"Frames should be 16, got {T}"
        assert H == 64, f"Height should be 64, got {H}"
        assert W == 64, f"Width should be 64, got {W}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

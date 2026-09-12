"""
Unit tests for Medallion architecture paths, dataset resolution, and MLflow lineage tracking.

Tests cover:
- New naming convention (0_landing, 1_bronze, 2_silver, 3_gold)
- Legacy fallback (00_landing, 01_bronze, 02_silver, 03_gold)
- Backward-compatible aliases (DATA == DATA_LAKE, CHECKPOINTS == TRAINED)
- Dynamic directory resolution preferring new names over legacy
"""

import json

from src.config.paths import (
    DATA,
    DATA_LAKE,
    LANDING,
    BRONZE,
    SILVER,
    GOLD,
    FEATURES,
    FEATURE_STORE,
    CHECKPOINTS,
    TRAINED,
    get_bronze_path,
    get_all_bronze_paths,
    get_silver_path,
    get_gold_path,
    _resolve_dir,
    _resolve_layer,
)
from src.data.gold import _resolve_gold_dir
from src.data.mlflow_tracker import (
    compute_file_sha256,
    create_gold_manifest,
    log_medallion_dataset,
)


class TestBackwardCompatibleAliases:
    """Verify backward-compatible aliases exist and point to the same paths."""

    def test_data_alias_equals_data_lake(self):
        """DATA and DATA_LAKE must resolve to the same path."""
        assert DATA == DATA_LAKE

    def test_checkpoints_alias_equals_trained(self):
        """CHECKPOINTS and TRAINED must resolve to the same path."""
        assert CHECKPOINTS == TRAINED

    def test_feature_store_alias(self):
        """FEATURE_STORE must equal FEATURES and live under SILVER."""
        assert FEATURE_STORE == FEATURES
        assert FEATURES.parent == SILVER


class TestDynamicResolution:
    """Test _resolve_dir and _resolve_layer helpers."""

    def test_resolve_dir_prefers_new_name(self, tmp_path):
        """When both 'data' and 'data_lake' exist, prefer 'data'."""
        (tmp_path / "data").mkdir()
        (tmp_path / "data_lake").mkdir()
        assert _resolve_dir(tmp_path, "data", "data_lake") == tmp_path / "data"

    def test_resolve_dir_falls_back_to_legacy(self, tmp_path):
        """When only 'data_lake' exists, use it."""
        (tmp_path / "data_lake").mkdir()
        assert _resolve_dir(tmp_path, "data", "data_lake") == tmp_path / "data_lake"

    def test_resolve_dir_defaults_to_new_name(self, tmp_path):
        """When neither exists, default to new name."""
        assert _resolve_dir(tmp_path, "data", "data_lake") == tmp_path / "data"

    def test_resolve_layer_prefers_new_prefix(self, tmp_path):
        """When both '0_landing' and '00_landing' exist, prefer '0_landing'."""
        (tmp_path / "0_landing").mkdir()
        (tmp_path / "00_landing").mkdir()
        assert _resolve_layer(tmp_path, "0", "00", "landing") == tmp_path / "0_landing"

    def test_resolve_layer_falls_back_to_legacy_prefix(self, tmp_path):
        """When only '00_landing' exists, use it."""
        (tmp_path / "00_landing").mkdir()
        assert _resolve_layer(tmp_path, "0", "00", "landing") == tmp_path / "00_landing"

    def test_resolve_layer_defaults_to_new_prefix(self, tmp_path):
        """When neither exists, default to new prefix."""
        assert _resolve_layer(tmp_path, "0", "00", "landing") == tmp_path / "0_landing"


class TestMedallionPaths:
    """Test path resolution for hierarchical and legacy Medallion layers."""

    def test_default_constants(self):
        """Layer names should use new convention OR legacy, depending on what exists on disk."""
        # These should be one of the expected names
        assert LANDING.name in ("0_landing", "00_landing")
        assert BRONZE.name in ("1_bronze", "01_bronze")
        assert SILVER.name in ("2_silver", "02_silver")
        assert GOLD.name in ("3_gold", "03_gold")
        assert FEATURES.name == "features"
        assert FEATURE_STORE == FEATURES
        assert FEATURES.parent == SILVER

    def test_get_bronze_path_hierarchical(self, monkeypatch, tmp_path):
        """When 1_bronze/<category> exists, it should be preferred."""
        data = tmp_path / "data"
        bronze_medical = data / "1_bronze" / "medical"
        bronze_medical.mkdir(parents=True)
        
        legacy_medical = data / "01_bronze_medical"
        legacy_medical.mkdir(parents=True)

        monkeypatch.setattr("src.config.paths.DATA", data)
        monkeypatch.setattr("src.config.paths.DATA_LAKE", data)
        monkeypatch.setattr("src.config.paths.BRONZE", data / "1_bronze")

        resolved = get_bronze_path("medical")
        assert resolved == bronze_medical

    def test_get_bronze_path_legacy_fallback(self, monkeypatch, tmp_path):
        """When 1_bronze/<category> does not exist but 01_bronze_<category> does, fall back."""
        data = tmp_path / "data"
        legacy_vision = data / "01_bronze_vision"
        legacy_vision.mkdir(parents=True)

        monkeypatch.setattr("src.config.paths.DATA", data)
        monkeypatch.setattr("src.config.paths.DATA_LAKE", data)
        monkeypatch.setattr("src.config.paths.BRONZE", data / "1_bronze")

        resolved = get_bronze_path("vision")
        assert resolved == legacy_vision

    def test_get_bronze_path_new_flat_legacy(self, monkeypatch, tmp_path):
        """When only '1_bronze_<category>' flat pattern exists, it should be found."""
        data = tmp_path / "data"
        flat_audio = data / "1_bronze_audio"
        flat_audio.mkdir(parents=True)

        monkeypatch.setattr("src.config.paths.DATA", data)
        monkeypatch.setattr("src.config.paths.DATA_LAKE", data)
        monkeypatch.setattr("src.config.paths.BRONZE", data / "1_bronze")

        resolved = get_bronze_path("audio")
        assert resolved == flat_audio

    def test_get_silver_and_gold_paths(self, monkeypatch, tmp_path):
        data = tmp_path / "data"
        silver_vis = data / "2_silver" / "vision"
        silver_vis.mkdir(parents=True)
        gold_vis = data / "3_gold" / "vision"
        gold_vis.mkdir(parents=True)

        monkeypatch.setattr("src.config.paths.SILVER", data / "2_silver")
        monkeypatch.setattr("src.config.paths.GOLD", data / "3_gold")

        assert get_silver_path("vision") == silver_vis
        assert get_silver_path() == data / "2_silver"
        assert get_gold_path("vision") == gold_vis
        assert get_gold_path() == data / "3_gold"

    def test_get_all_bronze_paths_hierarchical(self, monkeypatch, tmp_path):
        data = tmp_path / "data"
        bronze = data / "1_bronze"
        for cat in ["audio", "tabular", "vision"]:
            (bronze / cat).mkdir(parents=True)

        monkeypatch.setattr("src.config.paths.DATA", data)
        monkeypatch.setattr("src.config.paths.DATA_LAKE", data)
        monkeypatch.setattr("src.config.paths.BRONZE", bronze)

        paths = get_all_bronze_paths()
        assert len(paths) == 3
        assert sorted([p.name for p in paths]) == ["audio", "tabular", "vision"]

    def test_get_bronze_path_medical_alias_remapped(self, monkeypatch, tmp_path):
        data = tmp_path / "data"
        bronze = data / "1_bronze"
        bronze_vision = bronze / "vision"
        bronze_vision.mkdir(parents=True)

        monkeypatch.setattr("src.config.paths.DATA", data)
        monkeypatch.setattr("src.config.paths.DATA_LAKE", data)
        monkeypatch.setattr("src.config.paths.BRONZE", bronze)

        # When 1_bronze/medical does not exist, get_bronze_path("medical") resolves to 1_bronze/vision
        assert get_bronze_path("medical") == bronze_vision


class TestGoldResolution:
    """Test dataset resolution in Gold layer across hierarchical and flat layouts."""

    def test_resolve_flat_dataset(self, monkeypatch, tmp_path):
        gold = tmp_path / "3_gold"
        ds_train = gold / "my_dataset" / "train"
        ds_train.mkdir(parents=True)

        monkeypatch.setattr("src.data.gold.GOLD", gold)

        resolved = _resolve_gold_dir("my_dataset", "train")
        assert resolved == ds_train

    def test_resolve_hierarchical_dataset(self, monkeypatch, tmp_path):
        gold = tmp_path / "3_gold"
        ds_train = gold / "vision" / "kvasir_seg" / "train"
        ds_train.mkdir(parents=True)

        monkeypatch.setattr("src.data.gold.GOLD", gold)

        # Scoped with category
        resolved = _resolve_gold_dir("kvasir_seg", "train", category="vision")
        assert resolved == ds_train

        # Discovered automatically without category
        resolved_auto = _resolve_gold_dir("kvasir_seg", "train")
        assert resolved_auto == ds_train


class TestMLflowTracker:
    """Test SHA-256 calculation, manifest generation, and MLflow dataset lineage logging."""

    def test_compute_file_sha256(self, tmp_path):
        dummy_file = tmp_path / "test.txt"
        dummy_file.write_text("deep learning medallion test")

        sha256 = compute_file_sha256(dummy_file)
        assert isinstance(sha256, str)
        assert len(sha256) == 64

    def test_create_gold_manifest(self, tmp_path):
        archive = tmp_path / "kvasir_seg_v1.tar.gz"
        archive.write_bytes(b"dummy archive content")

        manifest_path = create_gold_manifest(
            archive_path=archive,
            dataset_name="kvasir_seg",
            category="vision",
            version="v1",
            split_strategy="group_aware_70_15_15",
            splits={"train": 800, "val": 100, "test": 100},
            source_bronze="1_bronze/vision/kvasir_seg",
            curation_recipe="scripts/data/curate_kvasir.py@commit_abc",
        )

        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)

        assert data["dataset_name"] == "kvasir_seg"
        assert data["category"] == "vision"
        assert data["version"] == "v1"
        assert data["split_strategy"] == "group_aware_70_15_15"
        assert data["splits"]["train"] == 800
        assert len(data["sha256"]) == 64

    def test_log_medallion_dataset_in_mlflow_run(self, tmp_path):
        import mlflow

        archive = tmp_path / "kvasir_seg_v1.tar.gz"
        archive.write_bytes(b"dummy archive content")
        manifest_path = create_gold_manifest(
            archive_path=archive,
            dataset_name="kvasir_seg",
            category="vision",
            version="v1",
            splits={"train": 800, "val": 100},
        )

        mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
        with mlflow.start_run():
            tags = log_medallion_dataset(
                archive_path=archive,
                manifest_path=manifest_path,
                context="training",
            )
            assert tags["medallion.tier"] == "3_gold"
            assert tags["medallion.dataset"] == "kvasir_seg"
            assert tags["medallion.category"] == "vision"
            assert tags["data.samples_train"] == "800"

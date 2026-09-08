"""Tests for reference-dataset setup and registration."""

from types import SimpleNamespace

from scripts import setup_datasets


def test_register_all_handles_serialized_partition_paths(tmp_path, monkeypatch, capsys):
    """Registry models serialize partition paths as strings."""
    config_path = tmp_path / "dataset.yaml"
    source_path = tmp_path / "source.csv"
    config_path.touch()
    source_path.touch()

    registered = []

    class FakeServices:
        def list_datasets(self):
            return registered

        def register_dataset_from_config(self, config, source):
            meta = SimpleNamespace(
                dataset_id="example",
                row_count=10,
                train_parquet_path="/tmp/example.train.parquet",
            )
            registered.append(meta)
            return meta

    monkeypatch.setattr(
        setup_datasets,
        "DATASETS",
        [(str(config_path), str(source_path))],
    )
    monkeypatch.setattr(setup_datasets, "get_services", lambda: FakeServices())

    from backend.datasets import config as config_module

    monkeypatch.setattr(
        config_module,
        "load_dataset_config",
        lambda path: SimpleNamespace(
            dataset_spec=SimpleNamespace(dataset_id="example")
        ),
    )

    setup_datasets.register_all(force=False)

    output = capsys.readouterr().out
    assert "train=example.train.parquet" in output
    assert "Total registered: 1 dataset(s)." in output


def test_generate_all_prints_named_progress(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "generated.parquet"
    commands = []
    monkeypatch.setattr(
        setup_datasets,
        "GENERATORS",
        [("Semiconductor memory supply history", "scripts/generate.py", str(output_path))],
    )
    monkeypatch.setattr(setup_datasets, "_run", commands.append)

    setup_datasets.generate_all(skip_gen=False)

    output = capsys.readouterr().out
    assert "First-time data generation may take a few minutes." in output
    assert "[1/1] Generating data: Semiconductor memory supply history" in output
    assert commands == [["python", "scripts/generate.py"]]

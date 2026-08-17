from pathlib import Path

import pytest

from core.config import ConfigError, load_config, load_configs


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_commented_optional_keyword_is_recognized_but_not_applied_by_default(tmp_path):
    default = load_config("stellar", CONFIG_DIR, run_dir=tmp_path)
    assert "bpass_metallicities" not in default.values

    (tmp_path / "config.dat").write_text("bpass_metallicities = 0001,0002,0008,0014\n")
    configured = load_config("stellar", CONFIG_DIR, run_dir=tmp_path)
    assert configured["bpass_metallicities"] == ["0001", "0002", "0008", "0014"]


def test_nested_excluded_intervals_parse_directly_from_config_dat(tmp_path):
    intervals = "[[900,1100],[1140,1147],[1180,1224],[1775,2998]]"
    (tmp_path / "config.dat").write_text(f"excluded_intervals = {intervals}\n")
    configs = load_configs(("stellar", "emission", "absorption"), CONFIG_DIR, run_dir=tmp_path)
    expected = [[900.0, 1100.0], [1140.0, 1147.0], [1180.0, 1224.0], [1775.0, 2998.0]]
    for config in configs.values():
        assert config["excluded_intervals"] == expected


def test_all_stellar_library_family_keywords_are_in_strict_schema(tmp_path):
    text = """\
stellar_library = bpass
bpass_h5_path = /tmp/bpass.h5
bpass_metallicities = 0001,0002
stellar_ages_myr = 1,2,3,5,10
stellar_distance_kpc = 4500.0
bpass_model_resolution_R = 20000.0
"""
    (tmp_path / "config.dat").write_text(text)
    cfg = load_config("stellar", CONFIG_DIR, run_dir=tmp_path)
    assert cfg["stellar_library"] == "bpass"
    assert cfg["bpass_h5_path"] == "/tmp/bpass.h5"
    assert cfg["stellar_ages_myr"] == [1, 2, 3, 5, 10]
    assert cfg["stellar_distance_kpc"] == pytest.approx(4500.0)


def test_optional_keywords_are_also_available_as_cli_overrides():
    cfg = load_config(
        "stellar", CONFIG_DIR,
        cli_args=["--bpass_metallicities", "0001,0008", "--stellar_distance_kpc", "1000"],
    )
    assert cfg["bpass_metallicities"] == ["0001", "0008"]
    assert cfg["stellar_distance_kpc"] == pytest.approx(1000.0)


def test_unknown_keyword_remains_fatal(tmp_path):
    (tmp_path / "config.dat").write_text("this_keyword_does_not_exist = 1\n")
    with pytest.raises(ConfigError):
        load_configs(("stellar", "emission", "absorption"), CONFIG_DIR, run_dir=tmp_path)


def test_launcher_keywords_are_optional_but_recognized(tmp_path):
    from core.config import load_configs
    config_dir = Path(__file__).resolve().parents[1] / "config"
    (tmp_path / "config.dat").write_text(
        "input_spectrum = /tmp/example.dat\n"
        "redshift = 0.001721\n"
        "mode = stellar\n"
    )
    configs = load_configs(("stellar", "emission", "absorption"), config_dir, run_dir=tmp_path)
    for config in configs.values():
        assert config.get("input_spectrum") == "/tmp/example.dat"
        assert config.get("redshift") == 0.001721
        assert config.get("mode") == "stellar"

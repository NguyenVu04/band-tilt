"""Tests for SUMO UE mobility generation — PROJECT.md section 9.1, Phase 2.

Unlike every other test module in this repository, most of this module is
**not** skipped. ``src/mobility/`` is implemented, not a stub, so most of its
pure logic — the coordinate transform, clipping, the trajectory schema, the
three invariant checks, scenario identity, and config composition — can be
tested without SUMO on ``PATH`` and without the DVC-tracked scene files. Only
the handful of tests that genuinely need one of those two things carry an
explicit ``@pytest.mark.skip``, each naming its own prerequisite.

The frame transform is the module this file is most careful about. An offset
error does not raise — it silently puts every UE in the wrong street — so
:func:`test_mapping_into_the_scene_frame_is_a_pure_translation` and its
neighbours check the arithmetic directly against
:func:`src.mobility.frame.SceneFrame` fixtures with round numbers, rather
than trusting the real UTM projection to be exercised only once, by eye, via
``task mobility:frame``.
"""

from typing import Any

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data import scenario
from src.mobility import checks, demand
from src.mobility import frame as frame_module
from src.mobility.simulate import tidy


def _override_mobility(cfg: DictConfig, section: str, **overrides: Any) -> DictConfig:
    """Return ``cfg`` with ``cfg.mobility.<section>`` merged with ``overrides``.

    A small helper rather than repeating the same nested-dict-merge at every
    call site: several tests need one changed key several levels deep in
    ``cfg.mobility`` while leaving everything else — including the other
    ``mobility`` sections — untouched.
    """
    updated_section = {**cfg.mobility[section], **overrides}
    return DictConfig({**cfg, "mobility": {**cfg.mobility, section: updated_section}})


# --- src.mobility.frame: the SUMO-network to Sionna-scene transform --------


def test_mapping_into_the_scene_frame_is_a_pure_translation(
    sumo_fcd_frame: pd.DataFrame, scene_frame: Any
) -> None:
    """``sim_x``/``sim_y`` shift by exactly ``(dx, dy)``; nothing else changes."""
    translated = frame_module.to_scene_frame(sumo_fcd_frame, scene_frame)
    pd.testing.assert_series_equal(
        translated["sim_x"], sumo_fcd_frame["sim_x"] + scene_frame.dx, check_names=False
    )
    pd.testing.assert_series_equal(
        translated["sim_y"], sumo_fcd_frame["sim_y"] + scene_frame.dy, check_names=False
    )
    pd.testing.assert_series_equal(translated["ue_id"], sumo_fcd_frame["ue_id"])
    pd.testing.assert_series_equal(translated["t"], sumo_fcd_frame["t"])


def test_mapping_into_the_scene_frame_round_trips(
    sumo_fcd_frame: pd.DataFrame, scene_frame: Any
) -> None:
    """Translating out and back by the inverse offset recovers the original."""
    inverse = frame_module.SceneFrame(
        dx=-scene_frame.dx,
        dy=-scene_frame.dy,
        bounds=scene_frame.bounds,
        net_offset=scene_frame.net_offset,
        proj_parameter=scene_frame.proj_parameter,
        scene_centre_lonlat=scene_frame.scene_centre_lonlat,
        utm_zone=scene_frame.utm_zone,
    )
    round_tripped = frame_module.to_scene_frame(
        frame_module.to_scene_frame(sumo_fcd_frame, scene_frame), inverse
    )
    pd.testing.assert_series_equal(round_tripped["sim_x"], sumo_fcd_frame["sim_x"])
    pd.testing.assert_series_equal(round_tripped["sim_y"], sumo_fcd_frame["sim_y"])


def test_positions_outside_the_scene_are_clipped_rather_than_asserted(
    sumo_fcd_frame: pd.DataFrame, scene_frame: Any, cfg: DictConfig
) -> None:
    """The one out-of-bounds sample (ue_2 @ t=20) is dropped under ``drop_samples``.

    Notebook 01's original section 3 check asserted every mapped position
    lies inside the scene bounds. That assertion is false against the
    delivered SUMO network, which extends beyond the Sionna-RT scene on all
    four sides (see ``configs/mobility.yaml``'s ``clip`` block) — so clipping,
    not asserting, is the correct behaviour, and this fixture is built to
    contain exactly one sample that needs it.
    """
    translated = frame_module.to_scene_frame(sumo_fcd_frame, scene_frame)
    clipped, stats = frame_module.clip_to_scene(translated, scene_frame, cfg)
    assert stats["rows_in"] == 12
    assert stats["rows_out"] == 11
    assert len(clipped) == 11
    assert not ((clipped["ue_id"] == "ue_2") & (clipped["t"] == 20.0)).any()


def test_the_clipped_fraction_is_reported_when_positions_fall_outside(
    sumo_fcd_frame: pd.DataFrame, scene_frame: Any, cfg: DictConfig
) -> None:
    """The clip stats report exactly 1/12, not just a pass/fail boolean."""
    translated = frame_module.to_scene_frame(sumo_fcd_frame, scene_frame)
    _, stats = frame_module.clip_to_scene(translated, scene_frame, cfg)
    assert stats["clipped_fraction"] == pytest.approx(1 / 12)


def test_clipping_whole_ues_removes_every_sample_of_an_offending_ue(
    sumo_fcd_frame: pd.DataFrame, scene_frame: Any, cfg: DictConfig
) -> None:
    """``drop_ues`` removes all four of ue_2's samples, not just the offending one."""
    cfg = _override_mobility(cfg, "clip", policy="drop_ues")
    translated = frame_module.to_scene_frame(sumo_fcd_frame, scene_frame)
    clipped, stats = frame_module.clip_to_scene(translated, scene_frame, cfg)
    assert stats["ues_dropped"] == 1
    assert (clipped["ue_id"] != "ue_2").all()
    assert len(clipped) == 8  # ue_1 and ue_3, four samples each


def test_clip_policy_error_raises_when_any_position_falls_outside(
    sumo_fcd_frame: pd.DataFrame, scene_frame: Any, cfg: DictConfig
) -> None:
    """``policy: error`` refuses to silently drop anything."""
    cfg = _override_mobility(cfg, "clip", policy="error")
    translated = frame_module.to_scene_frame(sumo_fcd_frame, scene_frame)
    with pytest.raises(ValueError, match="clip.policy='error'"):
        frame_module.clip_to_scene(translated, scene_frame, cfg)


def test_clipping_raises_when_more_than_the_configured_fraction_falls_outside(
    sumo_fcd_frame: pd.DataFrame, scene_frame: Any, cfg: DictConfig
) -> None:
    """A guard on the RESULT: too much clipping means the network and scene disagree."""
    cfg = _override_mobility(cfg, "clip", max_clipped_fraction=0.05)
    translated = frame_module.to_scene_frame(sumo_fcd_frame, scene_frame)
    with pytest.raises(ValueError, match="max_clipped_fraction"):
        frame_module.clip_to_scene(translated, scene_frame, cfg)


def test_the_derived_offset_is_rejected_when_it_moves_beyond_the_tolerance() -> None:
    """The tolerance guard fires when a fresh derivation drifts from the recorded one.

    Isolated from :func:`src.mobility.frame.derive_frame` on purpose: this is
    the one piece of that function that does not depend on a real net file or
    scene XML, so it is tested directly rather than only through the skipped
    end-to-end test below.
    """
    with pytest.raises(ValueError, match="tolerance_m"):
        frame_module.assert_offset_within_tolerance(
            dx=300.0, dy=129.4592, expected_dx=244.7237, expected_dy=129.4592, tolerance_m=1.0
        )


def test_the_derived_offset_is_accepted_within_the_tolerance() -> None:
    """A derivation that matches the recorded offset (within tolerance) does not raise."""
    frame_module.assert_offset_within_tolerance(
        dx=244.7238, dy=129.4591, expected_dx=244.7237, expected_dy=129.4592, tolerance_m=1.0
    )


@pytest.mark.skip(
    reason="needs the DVC-tracked scene.net.xml and scene.xml; run `task mobility:frame`"
)
def test_the_derived_offset_matches_the_delivered_network_and_scene(cfg: DictConfig) -> None:
    """The offset derived from the real files is ``(244.7237, 129.4592)`` metres.

    Hand-computed independently via a WGS84 -> UTM zone 48N forward
    projection, and confirmed against ``sumolib.net.Net.convertLonLat2XY`` on
    the real files — both agree to 5+ decimal places. Not runnable here
    because it needs ``data/external/simulation_map/scene.net.xml`` and
    ``scene.xml``, which are DVC-tracked and not part of a fresh clone.
    """
    frame = frame_module.derive_frame(cfg)
    assert frame.dx == pytest.approx(244.7237, abs=1e-3)
    assert frame.dy == pytest.approx(129.4592, abs=1e-3)


@pytest.mark.skip(reason="needs the DVC-tracked scene.net.xml")
def test_the_network_bounding_box_extends_beyond_the_scene_on_all_four_sides(
    cfg: DictConfig,
) -> None:
    """The delivered road network is larger than the ray-tracing scene, on every side.

    This is why :func:`test_positions_outside_the_scene_are_clipped_rather_than_asserted`
    exists: notebook 01's original "assert every position is inside the scene"
    check is false against the real data, not merely cautious.
    """
    from src.data import scenario as scenario_module
    from src.radio import scene

    frame = frame_module.derive_frame(cfg)
    location = scenario_module.read_net_location(cfg.mobility.network.net_file)
    conv = tuple(float(v) for v in location["convBoundary"].split(","))
    net_bbox_scene = (
        conv[0] + frame.dx,
        conv[1] + frame.dy,
        conv[2] + frame.dx,
        conv[3] + frame.dy,
    )
    sx0, sy0, sx1, sy1 = scene.scene_bounds(cfg)
    assert net_bbox_scene[0] < sx0
    assert net_bbox_scene[1] < sy0
    assert net_bbox_scene[2] > sx1
    assert net_bbox_scene[3] > sy1


@pytest.mark.skip(reason="needs SUMO on PATH; no test may shell out to a simulator")
def test_running_sumo_twice_with_one_seed_produces_identical_trajectories(cfg: DictConfig) -> None:
    """Determinism across the subprocess boundary: same seed, same trajectories.

    ``src.utils.seed.set_seed`` cannot reach a subprocess, so this property
    depends entirely on the three explicit seeds in
    :func:`src.mobility.demand.seeds` actually being passed through to
    ``randomTrips.py``, ``duarouter`` and ``sumo`` — worth a real end-to-end
    check, just not one any test in this suite may perform.
    """
    from src.mobility.simulate import generate

    first, _ = generate(cfg)
    second, _ = generate(cfg)
    pd.testing.assert_frame_equal(first, second)


# --- src.mobility.simulate.tidy ---------------------------------------------


def _raw_fcd(cfg: DictConfig) -> pd.DataFrame:
    """A tiny raw FCD frame in SUMO's own column naming, two UEs, unsorted."""
    return pd.DataFrame(
        {
            "timestep_time": [10.0, 0.0, 0.0],
            "vehicle_id": ["b", "a", "b"],
            "vehicle_x": [5.0, 0.0, 0.0],
            "vehicle_y": [5.0, 0.0, 0.0],
            "vehicle_speed": [1.0, 0.0, 0.0],
            "vehicle_angle": [90.0, 0.0, 0.0],
            "vehicle_type": ["ue_passenger", "ue_passenger", "ue_passenger"],
            "vehicle_lane": [":junc_2_0", "E1#3_0", "E1#3_0"],
        }
    )


def test_the_trajectory_frame_has_the_declared_columns_and_dtypes(
    cfg: DictConfig, scene_frame: Any
) -> None:
    """``tidy()`` produces exactly section 9.3's record, minus the radio columns."""
    out = tidy(_raw_fcd(cfg), cfg, "scn_test", scene_frame)
    assert out.columns.tolist() == [
        "scenario_id",
        "ue_id",
        "t",
        "date",
        "sim_x",
        "sim_y",
        "ue_height",
        "speed_mps",
        "heading_deg",
        "edge_id",
        "vehicle_type",
    ]
    assert out["scenario_id"].dtype == "string"
    assert out["ue_id"].dtype == "string"
    assert out["edge_id"].dtype == "string"
    assert str(out["date"].dtype) == "datetime64[ns, UTC]"
    assert (out["scenario_id"] == "scn_test").all()
    assert (out["ue_height"] == float(cfg.mobility.ue.height_m)).all()


def test_the_trajectory_frame_is_sorted_by_ue_and_time(cfg: DictConfig, scene_frame: Any) -> None:
    """Output order is ``(ue_id, t)``, not the order raw FCD rows happened to arrive in."""
    out = tidy(_raw_fcd(cfg), cfg, "scn_test", scene_frame)
    assert out["ue_id"].tolist() == ["a", "b", "b"]
    assert out["t"].tolist() == [0.0, 0.0, 10.0]


def test_the_edge_id_drops_only_the_trailing_lane_index(cfg: DictConfig, scene_frame: Any) -> None:
    """A lane id's LAST underscore-suffix is the lane index; the rest is the edge id.

    Junction-internal lanes (``":<junction>_<index>_<lane>"``) and ordinary
    edges (``"<edge>_<lane>"``) both have their edge id recovered by dropping
    only the final segment — this is what keeps an internal edge id like
    ``":junc_2"`` intact rather than truncated to ``":junc"``.
    """
    out = tidy(_raw_fcd(cfg), cfg, "scn_test", scene_frame)
    assert set(out["edge_id"]) == {":junc_2", "E1#3"}


def test_the_translation_matches_to_scene_frame(cfg: DictConfig, scene_frame: Any) -> None:
    """``tidy()`` applies the same translation as :func:`src.mobility.frame.to_scene_frame`."""
    out = tidy(_raw_fcd(cfg), cfg, "scn_test", scene_frame)
    row = out.loc[(out["ue_id"] == "a") & (out["t"] == 0.0)].iloc[0]
    assert row["sim_x"] == pytest.approx(0.0 + scene_frame.dx)
    assert row["sim_y"] == pytest.approx(0.0 + scene_frame.dy)


# --- src.mobility.checks: the three invariants ------------------------------


def test_the_speed_check_flags_a_sample_above_its_edge_limit(edge_speed_limits: Any) -> None:
    """Exactly the two rows above their edge's limit are flagged; the exact match is not."""
    df = pd.DataFrame(
        {
            "edge_id": ["edge_50", "edge_50", "edge_30", "edge_100"],
            "speed_mps": [13.89, 15.0, 8.33, 30.0],
        }
    )
    violations = checks.speed_within_limits(df, edge_speed_limits, cfg=None)
    assert violations.index.tolist() == [1, 3]
    assert violations["speed_limit_mps"].tolist() == [13.89, 27.78]


def test_the_speed_check_passes_when_nothing_exceeds_its_limit(edge_speed_limits: Any) -> None:
    """An empty result really means a clean pass, not an untested check."""
    df = pd.DataFrame({"edge_id": ["edge_50", "edge_30"], "speed_mps": [10.0, 5.0]})
    assert checks.speed_within_limits(df, edge_speed_limits, cfg=None).empty


def test_the_step_distance_check_flags_a_teleport(sumo_fcd_frame: pd.DataFrame) -> None:
    """The one designed-in 900 m / 10 s jump (ue_3) is flagged; nothing else is."""
    violations = checks.step_distance_is_consistent(sumo_fcd_frame, cfg=None)
    assert len(violations) == 1
    row = violations.iloc[0]
    assert row["ue_id"] == "ue_3"
    assert row["t"] == 10.0
    assert row["distance_m"] == pytest.approx(900.0)
    assert row["implied_speed_mps"] == pytest.approx(90.0)


def test_the_step_distance_check_does_not_flag_a_new_ue_first_sample(
    sumo_fcd_frame: pd.DataFrame,
) -> None:
    """A UE's first sample has no predecessor and must never be flagged as a jump."""
    violations = checks.step_distance_is_consistent(sumo_fcd_frame, cfg=None)
    first_samples = sumo_fcd_frame.groupby("ue_id")["t"].idxmin()
    assert not violations.index.isin(first_samples).any()


def test_the_arrival_check_compares_ue_count_over_time_with_the_configuration(
    cfg: DictConfig,
) -> None:
    """A bin with zero arrivals is flagged; bins matching the expected rate are not.

    Under the ``cfg`` fixture's window (100 s / 10 s bins, period=5 s ->
    2/bin expected).
    """
    times: list[float] = []
    for bucket in range(1, 10):  # bins [10,20) .. [90,100): two arrivals each
        t0 = bucket * 10 + 5
        times += [t0, t0]
    # bin [0, 10) is deliberately left with zero arrivals
    df = pd.DataFrame({"ue_id": [f"ue{i}" for i in range(len(times))], "t": times})
    out = checks.arrivals_match_configuration(df, cfg)
    assert len(out) == 1
    assert out.iloc[0]["bin_start"] == 0.0
    assert out.iloc[0]["observed"] == 0
    assert out.iloc[0]["expected"] == pytest.approx(2.0)


def test_run_all_collects_every_check_by_name(
    sumo_fcd_frame: pd.DataFrame, edge_speed_limits: Any, cfg: DictConfig
) -> None:
    """The dict returned by ``run_all`` has exactly the three check names as keys."""
    df = sumo_fcd_frame.assign(edge_id="edge_50", speed_mps=10.0)
    results = checks.run_all(df, edge_speed_limits, cfg)
    assert set(results) == {
        "speed_within_limits",
        "step_distance_is_consistent",
        "arrivals_match_configuration",
    }


# --- src.data.scenario: scenario_id -----------------------------------------


def _fingerprint(net_location_extra: str = "a") -> dict:
    return {
        "scene_file": "scene.xml",
        "scenegen": {"scenegen_UTM_zone": "EPSG:32648"},
        "net_file": "scene.net.xml",
        "net_location": {"netOffset": net_location_extra},
    }


def test_scenario_id_is_invariant_to_config_key_order(cfg: DictConfig) -> None:
    """Two configs with identical content in a different key order hash the same."""
    reordered = DictConfig(dict(reversed(list(cfg.mobility.items()))))
    cfg_a = DictConfig({**cfg})
    cfg_b = DictConfig({**cfg, "mobility": reordered})
    fp = _fingerprint()
    assert scenario.scenario_id(cfg_a, fp) == scenario.scenario_id(cfg_b, fp)


def test_scenario_id_changes_when_a_mobility_parameter_changes(cfg: DictConfig) -> None:
    """Changing ``ue.count`` changes the identity — it is part of the scenario definition."""
    fp = _fingerprint()
    baseline = scenario.scenario_id(cfg, fp)
    changed = _override_mobility(cfg, "ue", count=400)
    assert scenario.scenario_id(changed, fp) != baseline


def test_scenario_id_is_independent_of_the_output_directory(cfg: DictConfig) -> None:
    """A path is not part of what a scenario is; only where it happens to be written."""
    fp = _fingerprint()
    baseline = scenario.scenario_id(cfg, fp)
    changed = _override_mobility(cfg, "output", dir="somewhere/else")
    assert scenario.scenario_id(changed, fp) == baseline


def test_scenario_id_changes_when_the_network_location_changes(cfg: DictConfig) -> None:
    """A rebuilt network — different ``<location>`` — is a different environment."""
    fp_a = _fingerprint("a")
    fp_b = _fingerprint("b")
    assert scenario.scenario_id(cfg, fp_a) != scenario.scenario_id(cfg, fp_b)


def test_build_manifest_carries_scenario_id_and_every_section(cfg: DictConfig) -> None:
    """The manifest starts with identity, then carries every section verbatim."""
    manifest = scenario.build_manifest(cfg, "scn_test", counts={"n_rows": 42}, seeds={"trips": 1})
    assert manifest["scenario_id"] == "scn_test"
    assert manifest["counts"] == {"n_rows": 42}
    assert manifest["seeds"] == {"trips": 1}
    assert "created_utc" in manifest


# --- src.mobility.demand: seeds and departure period ------------------------


def test_the_three_subprocess_seeds_are_distinct_and_derived_from_one(cfg: DictConfig) -> None:
    """trips/route/sumo seeds are consecutive offsets from ``mobility.seed``, never equal."""
    seeds = demand.seeds(cfg)
    base = int(cfg.mobility.seed)
    assert seeds == {"trips": base, "route": base + 1, "sumo": base + 2}
    assert len(set(seeds.values())) == 3


def test_the_departure_period_follows_from_the_ue_count_and_the_run_window(
    cfg: DictConfig,
) -> None:
    """``period = (end_s - begin_s) / count`` — exactly, not approximately."""
    # cfg fixture: begin_s=0.0, end_s=100.0, ue.count=20
    assert demand.departure_period_s(cfg) == pytest.approx(5.0)


def test_departure_period_rejects_a_nonpositive_ue_count(cfg: DictConfig) -> None:
    """A count of zero or fewer UEs has no meaningful insertion period."""
    bad = _override_mobility(cfg, "ue", count=0)
    with pytest.raises(ValueError, match="count"):
        demand.departure_period_s(bad)


# --- Config composition ------------------------------------------------------


def test_the_ue_height_matches_the_radio_grid_height() -> None:
    """The real ``configs/mobility.yaml`` interpolates ``radio.grid.height_m``, not a copy.

    A measurement taken at a different height than the radio map is solved at
    is not comparable to it (see ``configs/mobility.yaml``'s ``ue`` block) —
    composes the real config files, not the ``cfg`` fixture, so a drift
    between the two is caught here rather than only by
    :func:`src.mobility.simulate.generate`'s runtime assertion.
    """
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        real_cfg = compose(config_name="config")
    assert real_cfg.mobility.ue.height_m == real_cfg.radio.grid.height_m


def test_the_mobility_config_composes_under_both_optimizer_groups() -> None:
    """``configs/mobility.yaml`` is present and unchanged by which optimizer is selected."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path("configs").resolve())
    for overrides in ([], ["optim=marl"]):
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            real_cfg = compose(config_name="config", overrides=overrides)
        assert "mobility" in real_cfg
        assert real_cfg.mobility.seed == real_cfg.seed

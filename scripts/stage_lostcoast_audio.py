#!/usr/bin/env python3
"""Stage CapFaces Lost Coast lean soundscape as a top-level ZIP.

Loose media/sc_*.wav on CapFaces hybrid discs push ISO9660 media/ past libogc's
reliable directory-find threshold (bootstrap.pk3 / gamecube.cfg inject flap).
Ship audio inside soundscape.pk3 at the gamedir root instead (same pattern as
extras.pk3 / G278 shallow VO).

Covers retail coast_lostcoast.util_birds / util_windgusts / util_crumblycliff
plus npc_seagull idle calls and windchimes used by the map's env_soundscapes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from vpk import open_vpk_roots, read_first  # noqa: E402

STREAM_BEDS = {
    "media/sc_shore.wav": ("ambient/water/wave1.wav", "0", "3.0"),
    "media/sc_distant.wav": ("ambient/water/distant_wave1.wav", "0", "3.0"),
}

# Tip-safe oneshots (0.5s mono). Order = playback cycle in soundscape_gamecube.c.
ONESHOTS = {
    # util_birds — full seagull + coastbird set from soundscapes_lostcoast.txt
    "media/sc_bird01.wav": "ambient/levels/coast/seagulls_ambient1.wav",
    "media/sc_bird02.wav": "ambient/levels/coast/seagulls_ambient2.wav",
    "media/sc_bird03.wav": "ambient/levels/coast/seagulls_ambient3.wav",
    "media/sc_bird04.wav": "ambient/levels/coast/seagulls_ambient4.wav",
    "media/sc_bird05.wav": "ambient/levels/coast/seagulls_ambient5.wav",
    "media/sc_bird06.wav": "ambient/levels/coast/coastbird1.wav",
    "media/sc_bird07.wav": "ambient/levels/coast/coastbird2.wav",
    "media/sc_bird08.wav": "ambient/levels/coast/coastbird3.wav",
    "media/sc_bird09.wav": "ambient/levels/coast/coastbird4.wav",
    "media/sc_bird10.wav": "ambient/levels/coast/coastbird5.wav",
    "media/sc_bird11.wav": "ambient/levels/coast/coastbird6.wav",
    "media/sc_bird12.wav": "ambient/levels/coast/coastbird7.wav",
    "media/sc_fly1.wav": "ambient/creatures/flies3.wav",
    "media/sc_fly2.wav": "ambient/creatures/flies5.wav",
    # npc_seagull creature calls (map has 5 npc_seagull)
    "media/sc_gull1.wav": "ambient/creatures/seagull_idle1.wav",
    "media/sc_gull2.wav": "ambient/creatures/seagull_idle2.wav",
    "media/sc_gull3.wav": "ambient/creatures/seagull_idle3.wav",
    # util_windgusts
    "media/sc_wind1.wav": "ambient/wind/wind_snippet3.wav",
    "media/sc_wind2.wav": "ambient/wind/wind_hit1.wav",
    "media/sc_wind3.wav": "ambient/wind/wind_hit2.wav",
    "media/sc_wind4.wav": "ambient/wind/wind_med1.wav",
    "media/sc_wind5.wav": "ambient/wind/wind_snippet4.wav",
    "media/sc_wind6.wav": "ambient/wind/wind_snippet5.wav",
    # util_crumblycliff (coast_lostcoast.cliffside)
    "media/sc_rock1.wav": "ambient/materials/rock2.wav",
    "media/sc_rock2.wav": "ambient/materials/rock3.wav",
    "media/sc_rock3.wav": "ambient/materials/rock5.wav",
    # oldwoodbuilding_with_windchimes
    "media/sc_chime1.wav": "ambient/levels/canals/windchine1.wav",
    "media/sc_chime2.wav": "ambient/levels/canals/windchime2.wav",
}


def vpk_roots(lostcoast: Path, hl2: Path):
    paths = sorted(lostcoast.glob("*_dir.vpk")) + sorted(hl2.glob("*_dir.vpk"))
    if not paths:
        raise SystemExit(f"no *_dir.vpk under {lostcoast} / {hl2}")
    return open_vpk_roots(paths)


def read_wav(roots, rel: str) -> bytes:
    data = read_first(roots, f"sound/{rel}") or read_first(roots, rel)
    if not data:
        raise SystemExit(f"missing sound/{rel} in VPKs")
    return data


def sox_convert(
    src: Path,
    dest: Path,
    *,
    trim: tuple[str, str] | None,
    rate: str,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["sox", str(src), "-r", rate, "-c", "1", "-b", "16", str(dest)]
    if trim:
        cmd.extend(["trim", trim[0], trim[1]])
    cmd.extend(["gain", "-n", "-1"])
    subprocess.run(cmd, check=True, capture_output=True)
    return dest.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--lostcoast", type=Path, required=True)
    ap.add_argument("--hl2", type=Path, required=True)
    args = ap.parse_args()

    stage: Path = args.stage
    stage.mkdir(parents=True, exist_ok=True)
    roots = vpk_roots(args.lostcoast, args.hl2)

    # Scrub prior loose audio that breaks CapFaces ISO9660 media/ finds.
    if (stage / "sound").exists():
        shutil.rmtree(stage / "sound")
    if (stage / "scripts").exists():
        shutil.rmtree(stage / "scripts")
    for stale in stage.glob("media/sc_*.wav"):
        stale.unlink()
    (stage / "audio-manifest.json").unlink(missing_ok=True)
    (stage / "soundscape.pk3").unlink(missing_ok=True)

    script_raw = read_first(roots, "scripts/soundscapes_lostcoast.txt")
    if script_raw:
        bake = ROOT / "cache" / "bake" / "scripts"
        bake.mkdir(parents=True, exist_ok=True)
        (bake / "soundscapes_lostcoast.txt").write_bytes(script_raw)

    tmp = stage / ".audio-tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    staged: dict[str, int] = {}

    for dest_rel, (src_rel, t0, tlen) in STREAM_BEDS.items():
        raw = tmp / Path(src_rel).name
        raw.write_bytes(read_wav(roots, src_rel))
        out = tmp / dest_rel.replace("/", "_")
        staged[dest_rel] = sox_convert(raw, out, trim=(t0, tlen), rate="16000")

    for dest_rel, src_rel in ONESHOTS.items():
        raw = tmp / Path(src_rel).name
        raw.write_bytes(read_wav(roots, src_rel))
        out = tmp / dest_rel.replace("/", "_")
        staged[dest_rel] = sox_convert(raw, out, trim=("0", "0.5"), rate="11025")

    pk3 = stage / "soundscape.pk3"
    # ZIP_STORED: CapFaces tip can't inflate (inflateInit2 -4 / Z_MEM_ERROR).
    with zipfile.ZipFile(pk3, "w", zipfile.ZIP_STORED) as zf:
        for dest_rel in list(STREAM_BEDS) + list(ONESHOTS):
            src = tmp / dest_rel.replace("/", "_")
            zf.write(src, dest_rel, compress_type=zipfile.ZIP_STORED)

    shutil.rmtree(tmp, ignore_errors=True)

    manifest = {
        "active_soundscape": "coast_lostcoast.general_shoreline",
        "shore_bed": "media/sc_shore.wav",
        "distant_bed": "media/sc_distant.wav",
        "pk3": "soundscape.pk3",
        "seagull_ambient": [k for k in ONESHOTS if "seagulls_ambient" in ONESHOTS[k]],
        "coastbird": [k for k in ONESHOTS if "coastbird" in ONESHOTS[k]],
        "seagull_idle": [k for k in ONESHOTS if "seagull_idle" in ONESHOTS[k]],
        "staged": staged,
        "bytes": sum(staged.values()),
        "pk3_bytes": pk3.stat().st_size,
        "files": len(staged),
    }
    (ROOT / "cache" / "bake" / "audio-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"lostcoast audio (pk3): {pk3.name}={manifest['pk3_bytes']} "
        f"wav_bytes={manifest['bytes']} files={len(staged)} "
        f"seagulls={len(manifest['seagull_ambient'])}+idle={len(manifest['seagull_idle'])} "
        f"coastbirds={len(manifest['coastbird'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

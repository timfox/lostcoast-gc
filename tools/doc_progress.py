#!/usr/bin/env python3
"""Document a Dolphin probe: still, retail side-by-side, MAE, short clip.

usage: tools/doc_progress.py <probe-dir> <label> [--note "what changed"]

Writes into the probe dir and mirrors to playtests/screenshots + videos:
  best.png                  last dumped frame
  compare-vs-retail.jpg     retail still | GameCube frame
  <label>.mp4               the dumped frames as a clip (when ffmpeg exists)
Appends one line to playtests/PROGRESS.md.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parents[1]
RETAIL = ROOT / "playtests/notes/gc-flipper-progress-20261109/retail-lostcoast-target.jpg"


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("probe", type=Path)
	ap.add_argument("label")
	ap.add_argument("--note", default="")
	args = ap.parse_args()
	frames = sorted(
		glob.glob(str(args.probe / "framedump_*.tga")),
		key=lambda f: int(f.rsplit("_", 1)[-1][:-4]),
	)
	if not frames:
		print("no framedumps in", args.probe, file=sys.stderr)
		return 1
	best = Image.open(frames[-1]).convert("RGB")
	best.save(args.probe / "best.png")
	retail = Image.open(RETAIL).convert("RGB").resize(best.size, Image.Resampling.LANCZOS)
	side = Image.new("RGB", (best.width * 2, best.height))
	side.paste(retail, (0, 0))
	side.paste(best, (best.width, 0))
	side.save(args.probe / "compare-vs-retail.jpg", quality=90)
	mae = sum(ImageStat.Stat(ImageChops.difference(retail, best)).mean) / 3.0

	shots = ROOT / "playtests/screenshots"
	vids = ROOT / "playtests/videos"
	shots.mkdir(parents=True, exist_ok=True)
	vids.mkdir(parents=True, exist_ok=True)
	shutil.copyfile(args.probe / "best.png", shots / f"{args.label}.png")
	shutil.copyfile(args.probe / "compare-vs-retail.jpg", shots / f"{args.label}-vs-retail.jpg")
	clip = ""
	if shutil.which("ffmpeg") and len(frames) > 1:
		out = vids / f"{args.label}.mp4"
		pattern = str(args.probe / "framedump_%d.tga")
		rc = subprocess.run(
			["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-start_number", "1",
			 "-i", pattern, "-pix_fmt", "yuv420p", "-vf", "scale=640:-2", str(out)],
		).returncode
		if rc == 0:
			# screenshots/ and videos/ are gitignored: keep a tracked copy in the probe dir.
			shutil.copyfile(out, args.probe / "clip.mp4")
			clip = f"notes/{args.probe.name}/clip.mp4"
	stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
	log = ROOT / "playtests/PROGRESS.md"
	if not log.exists():
		log.write_text("# Lost Coast GC render progress\n\n| When | Label | MAE vs retail | Still | Clip | Note |\n|---|---|---|---|---|---|\n")
	with log.open("a") as fh:
		fh.write(
			f"| {stamp} | {args.label} | {mae:.1f} | notes/{args.probe.name}/compare-vs-retail.jpg | {clip} | {args.note} |\n"
		)
	print(f"{args.label}: frames={len(frames)} mae={mae:.2f} clip={clip or 'none'}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

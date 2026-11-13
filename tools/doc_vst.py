#!/usr/bin/env python3
"""Document a video-stress-test (gcvst flythrough) probe.

usage: tools/doc_vst.py <probe-dir> <frames-dir> <label> [--note "..."]

The VST flythrough covers the whole map, so there is no single retail still
to diff against. Instead this records:
  vst.mp4            every dumped frame as a clip
  vst-sheet.jpg      contact sheet of 12 evenly spaced frames
  vst-stats.txt      per-frame luma / blown-white / near-black / unique-frame stats
and appends a row to playtests/PROGRESS.md (MAE column = "vst").
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("probe", type=Path)
	ap.add_argument("frames", type=Path)
	ap.add_argument("label")
	ap.add_argument("--note", default="")
	ap.add_argument("--fps", type=int, default=8)
	args = ap.parse_args()
	frames = sorted(
		glob.glob(str(args.frames / "framedump_*.tga")),
		key=lambda f: int(f.rsplit("_", 1)[-1][:-4]),
	)
	if not frames:
		print("no framedumps in", args.frames, file=sys.stderr)
		return 1
	args.probe.mkdir(parents=True, exist_ok=True)
	# The camera parks at the end of the ride while Dolphin keeps dumping:
	# trim the identical tail so the clip and sheet cover the ride itself.
	def _digest(path: str) -> str:
		return hashlib.md5(Path(path).read_bytes()).hexdigest()

	tail = _digest(frames[-1])
	keep = len(frames)
	while keep > 1 and _digest(frames[keep - 2]) == tail:
		keep -= 1
	frames = frames[: min(len(frames), keep + 8)]

	# Stats: how much of the run is distinct, blown out, or black.
	seen: set[str] = set()
	lines = ["frame luma white% black%"]
	lumas = []
	for i, f in enumerate(frames):
		im = Image.open(f).convert("RGB")
		seen.add(hashlib.md5(im.tobytes()).hexdigest())
		g = im.convert("L")
		hist = g.histogram()
		n = float(g.width * g.height)
		luma = ImageStat.Stat(g).mean[0]
		white = sum(hist[250:]) / n * 100.0
		black = sum(hist[:6]) / n * 100.0
		lumas.append(luma)
		lines.append(f"{i + 1} {luma:.1f} {white:.1f} {black:.1f}")
	summary = (
		f"frames={len(frames)} unique={len(seen)} "
		f"luma_mean={sum(lumas) / len(lumas):.1f} luma_min={min(lumas):.1f} luma_max={max(lumas):.1f}"
	)
	(args.probe / "vst-stats.txt").write_text(summary + "\n" + "\n".join(lines) + "\n")

	# Contact sheet: 12 evenly spaced frames, 4x3.
	picks = [frames[round(i * (len(frames) - 1) / 11)] for i in range(12)] if len(frames) >= 12 else frames
	tw, th = 320, 240
	cols = 4
	rows = (len(picks) + cols - 1) // cols
	sheet = Image.new("RGB", (cols * tw, rows * th))
	for i, f in enumerate(picks):
		im = Image.open(f).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
		sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
	sheet.save(args.probe / "vst-sheet.jpg", quality=88)

	clip = ""
	if shutil.which("ffmpeg") and len(frames) > 1:
		rc = subprocess.run(
			["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-start_number", "1",
			 "-i", str(args.frames / "framedump_%d.tga"), "-frames:v", str(len(frames)),
			 "-pix_fmt", "yuv420p",
			 "-vf", "scale=640:-2", str(args.probe / "vst.mp4")],
		).returncode
		if rc == 0:
			clip = f"notes/{args.probe.name}/vst.mp4"
	shots = ROOT / "playtests/screenshots"
	vids = ROOT / "playtests/videos"
	shots.mkdir(parents=True, exist_ok=True)
	vids.mkdir(parents=True, exist_ok=True)
	shutil.copyfile(args.probe / "vst-sheet.jpg", shots / f"{args.label}-sheet.jpg")
	if clip:
		shutil.copyfile(args.probe / "vst.mp4", vids / f"{args.label}.mp4")

	stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
	log = ROOT / "playtests/PROGRESS.md"
	with log.open("a") as fh:
		fh.write(
			f"| {stamp} | {args.label} | vst | notes/{args.probe.name}/vst-sheet.jpg | {clip} | {args.note} ({summary}) |\n"
		)
	print(f"{args.label}: {summary} clip={clip or 'none'}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

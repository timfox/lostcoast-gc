#!/usr/bin/env bash
# Probe out/lostcoastgc.iso in a private Dolphin profile and document it:
# still + retail side-by-side + clip + PROGRESS.md row (tools/doc_progress.py).
#   scripts/probe-doc.sh <label> ["note"]
# Uses its own Dolphin user dir so concurrent probes from other engineers /
# projects are not killed (probe-capfaces-smoke.sh pkills by user dir).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="${1:?label}"
NOTE="${2:-}"
export LCGC_DOLPHIN_USER="${LCGC_DOLPHIN_USER:-$HOME/.local/share/lostcoast-gc-claude/dolphin}"
SRC_USER="$HOME/.local/share/lostcoast-gc/dolphin"
if [[ ! -d "$LCGC_DOLPHIN_USER/Config" && -d "$SRC_USER/Config" ]]; then
	mkdir -p "$LCGC_DOLPHIN_USER"
	cp -a "$SRC_USER/Config" "$LCGC_DOLPHIN_USER/"
	[[ -d "$SRC_USER/GC" ]] && cp -a "$SRC_USER/GC" "$LCGC_DOLPHIN_USER/"
	sed -i "s|^DumpPath = .*|DumpPath = $LCGC_DOLPHIN_USER/Dump/Frames|" "$LCGC_DOLPHIN_USER/Config/GFX.ini"
fi
# Dolphin will not overwrite existing dumps — start from an empty dump dir.
mkdir -p "$LCGC_DOLPHIN_USER/Dump/Frames"
find "$LCGC_DOLPHIN_USER/Dump/Frames" -maxdepth 1 -name 'framedump_*' -delete
DOLPHIN_VIDEO="${DOLPHIN_VIDEO:-Vulkan}" DOLPHIN_BATCH=0 DOLPHIN_TIMEOUT="${DOLPHIN_TIMEOUT:-110}" \
	bash "$SCRIPT_DIR/probe-capfaces-smoke.sh" >"$LCGC_DOLPHIN_USER/last-probe.out" 2>&1 || true
# Other engineers probe concurrently: take our own logdir, not the newest one.
PROBE="$(awk -F= '/^logdir=/{print $2; exit}' "$LCGC_DOLPHIN_USER/last-probe.out")"
if [[ ! -d "$PROBE" && "${LCGC_DOC_VST:-0}" == "1" && -s "$LCGC_DOLPHIN_USER/Logs/dolphin.log" ]]; then
	# The smoke probe dies collecting thousands of VST dumps (head|pipefail):
	# keep our own notes dir from Dolphin's log instead.
	PROBE="$SCRIPT_DIR/../playtests/notes/vst-$(date +%Y%m%d-%H%M%S)"
	mkdir -p "$PROBE"
	cp -f "$LCGC_DOLPHIN_USER/Logs/dolphin.log" "$PROBE/dolphin.log"
fi
[[ -d "$PROBE" ]] || { echo "probe-doc: no logdir in probe output" >&2; exit 1; }
grep -h "bloom\|TEV material\|hot window\|alloc fail" "$PROBE/dolphin.log" | cut -c60-200 | sort | uniq -c | head -8 || true
if [[ "${LCGC_DOC_VST:-0}" == "1" ]]; then
	# Stress flythrough: document every dumped frame, not just the newest 40.
	grep -h "gcvst\|VST\|vst" "$PROBE/dolphin.log" | cut -c60-200 | sort | uniq -c | sort -rn | head -6 || true
	# Unlimited-speed runs can dump tens of thousands of 1 MB frames (34 GB
	# once): keep the ride itself, drop the parked tail and any torn last frame.
	MAXF="${LCGC_VST_MAX_FRAMES:-9000}"
	find "$LCGC_DOLPHIN_USER/Dump/Frames" -maxdepth 1 -name 'framedump_*.tga' \
		| awk -F_ -v m="$MAXF" '{n=$NF; sub(/\.tga$/,"",n); if (n+0>m) print}' | xargs -r rm -f
	find "$LCGC_DOLPHIN_USER/Dump/Frames" -maxdepth 1 -name 'framedump_*.tga' -size -900k -delete
	python3 "$SCRIPT_DIR/../tools/doc_vst.py" "$PROBE" "$LCGC_DOLPHIN_USER/Dump/Frames" "$LABEL" --note "$NOTE"
	# The clip and sheet are saved: free the raw dumps.
	find "$LCGC_DOLPHIN_USER/Dump/Frames" -maxdepth 1 -name 'framedump_*' -delete
else
	python3 "$SCRIPT_DIR/../tools/doc_progress.py" "$PROBE" "$LABEL" --note "$NOTE"
fi
echo "probe=$PROBE"

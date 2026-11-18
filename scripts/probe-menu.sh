#!/usr/bin/env bash
# Dolphin probe: retail title menu → New Game → d2_lostcoast (LCG-7).
# Expects a --menu ISO (gamecube.cfg has startmap + menuauto, no map=).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

ISO="${1:-$LCGC_ISO}"
[[ -f "$ISO" ]] || { echo "error: missing ISO $ISO" >&2; exit 1; }

DOLPHIN_USER="${LCGC_DOLPHIN_USER:-$HOME/.local/share/lostcoast-gc/dolphin}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="$LCGC_NOTES/dolphin-probe-menu-$STAMP"
mkdir -p "$DOLPHIN_USER" "$LOGDIR" "$LCGC_SHOTS"

# Confirm ISO is a menu boot before spending Dolphin time.
python3 - <<PY
from pathlib import Path
import subprocess, tempfile, os, sys
iso = Path("$ISO")
vtd = Path(tempfile.mkdtemp(prefix="lcgc-menu-check-"))
try:
	for rel in ("lostcoast/gamecube.cfg", "lostcoast/liblist.gam", "lostcoast/resource/gc_menu/SOURCE.txt"):
		dest = vtd / Path(rel).name
		subprocess.check_call(
			["xorriso", "-indev", str(iso), "-osirrox", "on", "-extract", f"/xash3d/{rel}", str(dest)],
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
		)
	cfg = (vtd / "gamecube.cfg").read_text()
	lib = (vtd / "liblist.gam").read_text()
	src = (vtd / "SOURCE.txt").read_text()
	cfg_lines = [ln.strip() for ln in cfg.splitlines() if ln.strip() and not ln.strip().startswith("#")]
	if any(ln == "map d2_lostcoast" or ln.startswith("map d2_lostcoast ") for ln in cfg_lines) or "gccapfacessmoke" in cfg:
		raise SystemExit("ISO is CapFaces/VST smoke, not --menu (rebuild with --menu)")
	if "startmap d2_lostcoast" not in cfg and "startmap" not in lib:
		raise SystemExit("ISO missing startmap d2_lostcoast")
	if "Lost Coast" not in src:
		raise SystemExit("ISO gc_menu is not Lost Coast title bake")
	print("menu ISO check: OK")
	print(cfg)
finally:
	for p in vtd.iterdir():
		p.unlink(missing_ok=True)
	vtd.rmdir()
PY

lcgc_pkill_own_dolphin "$DOLPHIN_USER"

if [[ -z "${DOLPHIN_BIN:-}" ]]; then
	if [[ -x "$LCGC_ENGINE/3rdparty/dolphin/build/Binaries/xash-gc-dolphin" ]]; then
		DOLPHIN_BIN="$LCGC_ENGINE/3rdparty/dolphin/build/Binaries/xash-gc-dolphin"
	elif [[ -x "$LCGC_ENGINE/3rdparty/dolphin/build/Binaries/dolphin-emu" ]]; then
		DOLPHIN_BIN="$LCGC_ENGINE/3rdparty/dolphin/build/Binaries/dolphin-emu"
	elif command -v dolphin-emu >/dev/null 2>&1; then
		DOLPHIN_BIN="$(command -v dolphin-emu)"
	elif command -v dolphin-emu-nogui >/dev/null 2>&1; then
		DOLPHIN_BIN="$(command -v dolphin-emu-nogui)"
	else
		echo "error: no Dolphin found" >&2
		exit 1
	fi
fi

mkdir -p "$DOLPHIN_USER/Config" "$DOLPHIN_USER/Logs"
cat > "$DOLPHIN_USER/Config/Logger.ini" <<'EOF'
[Options]
WriteToFile = True
WriteToConsole = True
Verbosity = 4
[Logs]
OSREPORT = True
BOOT = True
OSREPORT_HLE = True
EOF
# Truncate so a prior CapFaces/VST log cannot fake MAP READY gates.
: > "$DOLPHIN_USER/Logs/dolphin.log"

# Menu dwell + New Game + map load needs more than CapFaces smoke.
TIMEOUT="${DOLPHIN_TIMEOUT:-180}"
VIDEO_BACKEND="${DOLPHIN_VIDEO:-Null}"
BATCH_ARGS=(-b -v "$VIDEO_BACKEND")
echo "==> Dolphin menu probe ($TIMEOUT s) bin=$(basename "$DOLPHIN_BIN") user=$DOLPHIN_USER"

set +e
timeout --signal=KILL "$TIMEOUT" "$DOLPHIN_BIN" "${BATCH_ARGS[@]}" \
	-u "$DOLPHIN_USER" -e "$ISO" \
	>"$LOGDIR/stdout.log" 2>"$LOGDIR/stderr.log"
RC=$?
set -e

cp -f "$DOLPHIN_USER/Logs/dolphin.log" "$LOGDIR/dolphin.log" 2>/dev/null || true
{
	echo "# Lost Coast menu probe $STAMP"
	echo "- ISO: $ISO"
	echo "- timeout: $TIMEOUT"
	echo "- dolphin_rc: $RC"
	echo
	echo "## OSReport matches"
	grep -Eih 'main menu opened|gc_menu_ready|menuauto|disc boot override menu|play start|startmap|world DL|MAP READY|MAP LOADED|New Game|Lost Coast|title' \
		"$LOGDIR/dolphin.log" "$LOGDIR/stderr.log" 2>/dev/null | head -80 || true
} > "$LOGDIR/EVIDENCE.md"

TITLE=0
MENU_OPEN=0
PLAY_START=0
MAP_READY=0
# Lost Coast title menu: gc_menu_ready + title tiles (mainui may not print "main menu opened").
grep -Eiq 'main menu opened|gc_menu_ready set 1|menu tile resource/gc_menu/' \
	"$LOGDIR/dolphin.log" "$LOGDIR/stderr.log" 2>/dev/null && MENU_OPEN=1
grep -Eiq 'gc_menu_ready set 1|probe menu ready gate open|probe menuauto New Game dwell' \
	"$LOGDIR/dolphin.log" "$LOGDIR/stderr.log" 2>/dev/null && TITLE=1
grep -Eiq 'play start begin d2_lostcoast|play start uses test start map d2_lostcoast|probe menuauto New Game force gc_playstart' \
	"$LOGDIR/dolphin.log" "$LOGDIR/stderr.log" 2>/dev/null && PLAY_START=1
grep -Eiq 'world DL Active load map=d2_lostcoast|direct map ready|play start post-activate d2_lostcoast|MAP READY|frame budget samples armed after map ready' \
	"$LOGDIR/dolphin.log" "$LOGDIR/stderr.log" 2>/dev/null && MAP_READY=1

{
	echo
	echo "## Gates"
	echo "- menu_opened=$MENU_OPEN"
	echo "- title_dwell=$TITLE"
	echo "- play_start=$PLAY_START"
	echo "- map_ready=$MAP_READY"
} >> "$LOGDIR/EVIDENCE.md"

echo "MENU_PROBE: menu_opened=$MENU_OPEN title_dwell=$TITLE play_start=$PLAY_START map_ready=$MAP_READY ($LOGDIR)"
if (( MENU_OPEN && TITLE && PLAY_START && MAP_READY )); then
	echo "MENU_PROOF: PASS"
	exit 0
fi
if (( MENU_OPEN && TITLE )); then
	echo "MENU_PROOF: PARTIAL — title menu OK, map load incomplete"
	exit 2
fi
echo "MENU_PROOF: FAIL"
exit 1

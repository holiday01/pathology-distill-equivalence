#!/usr/bin/env bash
# Re-run the two stage-5 scripts and the H0-mini audit against the 270-slide
# bundle. Both stages failed in the pipeline run because the runtime-rewritten
# copy was placed outside scripts/, and those files do
# sys.path.insert(0, Path(__file__).parent) to reach their sibling modules.
# The copy now lives in scripts/ and is removed afterwards.
set -u
cd "$(dirname "$0")/.."

H5=/path/to/cache/patches/patches_c16_annotated_270.h5
RES=outputs/v4_full/c16_270

for s in compute_intra_slide_icc compute_c16_calibration_annotated; do
  echo "=== $s ==="
  PIPE_H5="$H5" PIPE_ROOT="$RES" PIPE_SCRIPT="scripts/$s.py" python3 - <<'PY'
import os, re, pathlib, runpy, sys
src = pathlib.Path(os.environ["PIPE_SCRIPT"]).read_text()
h5, root = os.environ["PIPE_H5"], os.environ["PIPE_ROOT"]
src = re.sub(r'^(C16|H5)\s*=\s*".*"$', lambda m: f'{m.group(1)} = "{h5}"', src, flags=re.M)
src = re.sub(r'^ROOT\s*=\s*".*"$', f'ROOT = "{root}"', src, flags=re.M)
src = src.replace('"outputs/v4_full/intra_slide_icc.json"', f'"{root}/intra_slide_icc.json"')
assert h5 in src and root in src, "path rewrite failed"
tmp = pathlib.Path("scripts") / (pathlib.Path(os.environ["PIPE_SCRIPT"]).stem + "_270_tmp.py")
tmp.write_text(src)
try:
    sys.argv = [str(tmp)]
    runpy.run_path(str(tmp), run_name="__main__")
finally:
    tmp.unlink(missing_ok=True)
PY
done

echo "=== h0mini vs H-Optimus-0 ==="
H5_ANNOT="$H5" python3 - <<'PY'
import os, re, pathlib, runpy, sys
src = pathlib.Path("scripts/h6_flip_h0mini.py").read_text()
src = re.sub(r'^H5 = ".*"$', f'H5 = "{os.environ["H5_ANNOT"]}"', src, flags=re.M)
assert os.environ["H5_ANNOT"] in src, "path rewrite failed"
tmp = pathlib.Path("scripts/_h0mini_270_tmp.py")
tmp.write_text(src)
try:
    sys.argv = [str(tmp)]
    runpy.run_path(str(tmp), run_name="__main__")
finally:
    tmp.unlink(missing_ok=True)
PY

echo "=== done ==="

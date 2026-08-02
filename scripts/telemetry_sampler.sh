#!/usr/bin/env bash
# Samples nvidia-smi every 10s into a CSV for paper-ready memory/power/thermal table.
# Usage: telemetry_sampler.sh <output_csv> [interval_sec]
set -u
OUT="${1:?need output csv}"
INTERVAL="${2:-10}"
# header
if [ ! -f "$OUT" ]; then
  echo "ts,gpu_util_pct,mem_used_mib,mem_total_mib,temp_c,power_w,fan_pct,clock_graphics_mhz,clock_mem_mhz" > "$OUT"
fi
while true; do
  TS=$(date -Iseconds)
  ROW=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed,clocks.gr,clocks.mem \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  echo "${TS},${ROW}" >> "$OUT"
  sleep "$INTERVAL"
done

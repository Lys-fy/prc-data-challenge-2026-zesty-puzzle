#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Member gate of a MODE=new workspace. Every production member prediction of the line must be a LOCAL file of the workspace (written on
# this period's features), not a link to another ranking set's predictions. Prints the missing list; with MARK=1 and nothing missing it
# writes the phase2 markers (prep, m_<family> of phase2/manifest/members.tsv except PSEUDOLABEL, m_XNCLK, m_BIG). K250, KCATLg and KSTK have
# their own markers (scripts/kumo_prod.sh).
#   usage: [MARK=1] bash scripts/members_check.sh <ws_name>        root = $PRC_ROOT, else the current directory
set -uo pipefail; cd "${PRC_ROOT:-$PWD}"; WSN=${1:?ws}; WS=phase2/work/$WSN; D=$WS/data/preds; miss=0; ok=0
chk(){ if [ -f "$1" ] && [ ! -L "$1" ]; then ok=$((ok+1)); else echo "MISSING/LINK $1"; miss=$((miss+1)); fi; }
while IFS=$'\t' read -r fam rng rest; do
  case $fam in \#*|""|pseudolabel_*) continue;; esac
  lo=${rng#s}; lo=${lo%-*}; hi=${rng#*-}
  for s in $(seq $lo $hi); do chk $D/${fam}_s$s.parquet; done
done < <(awk -F'\t' '!/^#/ && NF > 5' phase2/manifest/members.tsv)
for f in nbr_clock_mbmp_s0 nbr_clock_map_s0 big_bigp_s0 big_bigp_s1 big_bigp_s2; do chk $D/$f.parquet; done
echo "member gate $WSN: $ok local, $miss missing or still linked"
if [ "$miss" = 0 ] && [ "${MARK:-0}" = 1 ]; then
  touch $WS/logs/prep.done $WS/logs/m_XNCLK.done $WS/logs/m_BIG.done
  for fam in $(awk -F'\t' '!/^#/ && NF > 5 && $1 !~ /^pseudolabel_/ {print $1}' phase2/manifest/members.tsv); do touch $WS/logs/m_$fam.done; done
  echo "markers written: prep, m_<family> (members.tsv), m_XNCLK, m_BIG"
fi
[ "$miss" = 0 ]

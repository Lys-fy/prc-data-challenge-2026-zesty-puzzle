# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail; cd ${PRC_ROOT:-$PWD}
pru=1 bash scripts/chain_stk.sh KBNNA3
FORM=PC A1=1 bash scripts/chain_stage.sh scratchpad/p13/P13_KBNNA3xTD KBNNA3xca

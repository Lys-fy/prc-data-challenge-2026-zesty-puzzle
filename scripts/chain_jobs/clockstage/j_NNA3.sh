# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail; cd ${PRC_ROOT:-$PWD}
clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=4 bash scripts/clockstage_fold.sh scratchpad/p13/P13_KBNNA3xTD LNNA3

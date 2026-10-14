# SPDX-License-Identifier: GPL-3.0-only
# Clock stages of chain dTK with the longest-stop clocks: extraction (clkL, 410 days) -> adsbL table -> fold clock stages (push-start form + adsbL)
# -> comparison with the chain without adsbL (pushstart_xpb6dtk_ha0l, month-weighted rule)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; X=scratchpad/xau1
NP=4 $P scripts/adsb_longstop_extract.py --days $X/days_all.txt | tail -3
$P scripts/build_adsbL_features.py | head -3
clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=4 bash scripts/clockstage_fold.sh scratchpad/p13/P13_KBdTKxTD LdTK
$P scripts/month_weight_rescore.py --mwd screen_LdTK~pushstart_clockdtk=npy:$X/clockfold1_ldtk_ha0l~npy:scratchpad/xpb/pushstart_xpb6dtk_ha0l info_LdTK~T2xdTK=npy:$X/clockfold1_ldtk_ha0l~npy:scratchpad/p13/P13_KBdTKxca_HA0L

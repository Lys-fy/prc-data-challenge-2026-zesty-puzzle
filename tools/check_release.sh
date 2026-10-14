#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Pre-publication scan of the release tree. Exit 1 if anything that must not be published is found:
#   1. data files (competition rows, derived per-row tables, OOF, predictions, submissions, models)
#   2. large files (> 1 MB) of any type
#   3. secrets and credentials (keys, tokens, private keys, credential files)
#   4. host-specific paths (home directories, private IPs) and private names; a warning only
# usage: [CHECK_RELEASE_PRIVATE='name1|name2|...'] bash tools/check_release.sh [release_root]
#   CHECK_RELEASE_PRIVATE: extra extended-regex alternatives (user names, host nicknames, project roots) that must not appear.
#   Give them on the command line only, so the private names are never stored in the repository.
set -uo pipefail
R=${1:-.}; cd "$R" || exit 2; fail=0
note(){ echo "[$1] $2"; }

# 1. data files by extension or name
D=$(find . -type f \( -iname '*.parquet' -o -iname '*.npz' -o -iname '*.npy' -o -iname '*.feather' -o -iname '*.arrow' \
     -o -iname '*.pkl' -o -iname '*.pickle' -o -iname '*.joblib' -o -iname '*.cbm' -o -iname '*.lgb' -o -iname '*.pt' -o -iname '*.pth' \
     -o -iname '*.h5' -o -iname '*.tar*' -o -iname '*.zip' -o -iname '*.gz' -o -iname '*.xlsx' -o -iname '*.dat' -o -iname '*.jsonl' \
     -o -iname 'zesty-puzzle_*' \) -not -path './.git/*')
[ -n "$D" ] && { note FAIL "data / model / archive files:"; echo "$D"; fail=1; } || note ok "no data, model or archive files"
C=$(find . -type f -iname '*.csv' -not -path './.git/*'); [ -n "$C" ] && { note FAIL "csv files (check by hand):"; echo "$C"; fail=1; }

# 2. large files
L=$(find . -type f -size +1M -not -path './.git/*'); [ -n "$L" ] && { note FAIL "files > 1 MB:"; echo "$L"; fail=1; } || note ok "no file > 1 MB"

# 3. secrets
F=$(find . -type f \( -iname 'credentials*' -o -iname '*secret*' -o -iname '.env' -o -iname '*.pem' -o -iname '*.key' -o -iname 'id_rsa*' \
     -o -iname '*.netrc' -o -iname 'access.json' \) -not -path './.git/*' -not -name 'check_release.sh')
[ -n "$F" ] && { note FAIL "credential-like file names:"; echo "$F"; fail=1; } || note ok "no credential-like file names"
P='(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}|xox[abpr]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|[MN][A-Za-z0-9_-]{23}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}|(access|secret|api|client)_?(key|secret|token)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][A-Za-z0-9/+]{12,}|pass(word|wd)?["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{4,})'
S=$(grep -rIEin "$P" . --exclude=check_release.sh --exclude-dir=.git 2>/dev/null)
[ -n "$S" ] && { note FAIL "secret-like strings:"; echo "$S"; fail=1; } || note ok "no secret-like strings"
E=$(grep -rIEn '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}' . --exclude=check_release.sh --exclude-dir=.git --exclude=LICENSE 2>/dev/null | grep -v 'noreply@\|fsf.org')
[ -n "$E" ] && { note WARN "e-mail addresses (check):"; echo "$E"; }

# 4. host paths, private addresses and private names (warning). Code reads its root from $PRC_ROOT (fallback: the current directory).
HP='/(home|Users)/[A-Za-z0-9_.-]+|/root/[A-Za-z0-9_.-]+|~/[A-Za-z0-9_.-]+-20[0-9][0-9]'
[ -n "${CHECK_RELEASE_PRIVATE:-}" ] && HP="$HP|$CHECK_RELEASE_PRIVATE"
IP='(^|[^0-9.])(10\.[0-9]+|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+([^0-9.]|$)'   # private IPv4; package pin files skipped (versions look alike)
H=$( { grep -rIEl "$HP" . --exclude=check_release.sh --exclude-dir=.git 2>/dev/null
       grep -rIEl "$IP" . --exclude=check_release.sh --exclude='requirements*.txt' --exclude='licences-*.tsv' --exclude='*.lock*' --exclude-dir=.git 2>/dev/null; } | sort -u)
[ -n "$H" ] && { note WARN "host paths / private IPs / private names remain in $(echo "$H" | wc -l) files:"; echo "$H"; } || note ok "no host paths or private names"

# 5. private names given on the command line (case-insensitive): FAIL. The HTTP header name "User-Agent" of the two download scripts is not a name.
if [ -n "${CHECK_RELEASE_PRIVATE:-}" ]; then
  N=$(grep -rIEin "$CHECK_RELEASE_PRIVATE" . --exclude=check_release.sh --exclude-dir=.git 2>/dev/null | grep -v '"User-Agent"')
  [ -n "$N" ] && { note FAIL "private names: $(echo "$N" | wc -l) lines"; echo "$N" | cut -c1-200; fail=1; } || note ok "no private names ($(echo "$CHECK_RELEASE_PRIVATE" | tr '|' ' ' | wc -w) patterns)"
fi
[ $fail = 0 ] && echo "RESULT: clean" || echo "RESULT: NOT CLEAN"
exit $fail

#!/bin/bash
# Downloads the FLAME2020 model, which requires a free registration at
# https://flame.is.tue.mpg.de/ and cannot be fetched automatically.
#
# Run this script YOURSELF from a shell on Leonardo (e.g. via the `!` prefix
# in Claude Code, or directly in a terminal). It prompts for your FLAME
# credentials interactively (read -p / -s) so they are never typed into
# a chat message, never appear in a Claude Code transcript, and are not
# stored in your shell history as command-line arguments.
#
# Usage:
#   cd /leonardo_work/IscrC_SLPSCALE/TEASER
#   bash scripts/download_flame.sh

set -e

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

cd "$(dirname "$0")/.."

echo "If you do not have an account, register at https://flame.is.tue.mpg.de/"
read -p "Username (FLAME): " username
read -s -p "Password (FLAME): " password
echo
username=$(urle "$username")
password=$(urle "$password")

echo "Downloading FLAME2020..."
mkdir -p assets/FLAME2020
wget --post-data "username=$username&password=$password" \
     'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
     -O './FLAME2020.zip' --no-check-certificate --continue
unzip -o FLAME2020.zip -d assets/FLAME2020/
rm FLAME2020.zip

if [ -f assets/FLAME2020/generic_model.pkl ]; then
    echo "Success: assets/FLAME2020/generic_model.pkl is in place."
else
    echo "generic_model.pkl not found at the top level, checking archive layout..."
    find assets/FLAME2020 -iname "generic_model.pkl"
fi

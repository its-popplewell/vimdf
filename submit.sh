#!/usr/bin/env bash
# Submit the shard array. Logs go to this repo's logs/ directory.
#
# usage: ./submit.sh <project-code>
#    or: VIMDF_ACCOUNT=<project-code> ./submit.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${1:-${VIMDF_ACCOUNT:-}}"

if [ -z "$ACCOUNT" ]; then
    echo "usage: ./submit.sh <project-code>" >&2
    exit 2
fi
if [ ! -x "$REPO/.venv/bin/python" ]; then
    echo "error: no venv at $REPO/.venv; run ./setup.sh first" >&2
    exit 1
fi

mkdir -p "$REPO/logs"
cd "$REPO"    # sets PBS_O_WORKDIR, which the job script cd's into
qsub -A "$ACCOUNT" -o "$REPO/logs/" jobs/vimdf_shards.pbs
echo "logs will be written to $REPO/logs/"

# Common settings for the server scripts (sourced, not run).
#
# Python runs inside the sbwm container (docker-compose.yml) with the
# repository mounted at /work; git runs on the host with the server's key.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

#Lorenzo Giannuzzo: with rootless Docker the container root is the host user,
#so files written in /work stay owned by us; otherwise the container runs with
#our own uid and gid.
if docker info --format '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless; then
    export HOST_UID=0 HOST_GID=0
else
    export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
fi

WORKERS="${WORKERS:-28}"   # simulations in parallel (32 cores, a few left for the system)
THREADS="${THREADS:-16}"   # torch threads for world-model and policy training

py() { docker compose run --rm -T sbwm python "$@"; }

push_results() {
    # push_results "message" path [path ...]
    local msg="$1"; shift
    git add "$@"
    git commit -m "$msg" || true
    git pull --rebase origin main
    git push origin main
}

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_ROOT="${REPO_ROOT}/openr1"

if [[ ! -d "${ENV_ROOT}" ]]; then
  echo "[conda-hook] ${ENV_ROOT} not found. Run 'make conda-local' first." >&2
  exit 1
fi

ACTIVATE_DIR="${ENV_ROOT}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_ROOT}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/openr1-local.sh" <<EOF
#!/usr/bin/env bash
OPENR1_ROOT="${REPO_ROOT}"
OPENR1_LOCAL_ENV_SILENT=1
if [ -f "\$OPENR1_ROOT/tools/local_env_body.sh" ]; then
  . "\$OPENR1_ROOT/tools/local_env_body.sh"
fi
EOF

cat > "${DEACTIVATE_DIR}/openr1-local.sh" <<EOF
#!/usr/bin/env bash
if [ -f "${REPO_ROOT}/tools/local_env_clear.sh" ]; then
  . "${REPO_ROOT}/tools/local_env_clear.sh"
fi
EOF

chmod +x "${ACTIVATE_DIR}/openr1-local.sh" "${DEACTIVATE_DIR}/openr1-local.sh"
echo "[conda-hook] Installed local cache activation scripts."

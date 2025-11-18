#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script: source tools/local_env.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENR1_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export OPENR1_ROOT

source "${SCRIPT_DIR}/local_env_body.sh"

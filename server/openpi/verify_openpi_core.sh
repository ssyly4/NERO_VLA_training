#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT="15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_ROOT="${1:-/home/dev/workspace/openpi_deploy/repos/openpi}"

git -C "$OPENPI_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "[FAIL] OpenPI git worktree not found: $OPENPI_ROOT" >&2
  exit 2
}
[[ "$(git -C "$OPENPI_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "[FAIL] OpenPI is not based on $EXPECTED_COMMIT" >&2
  exit 2
}

required=(
  src/openpi/models/rtc.py
  src/openpi/policies/nero_policy.py
  src/openpi/policies/nero_bimanual_policy.py
)
for path in "${required[@]}"; do
  [[ -f "$OPENPI_ROOT/$path" ]] || {
    echo "[FAIL] missing NERO core file: $path" >&2
    exit 2
  }
done

cd "$OPENPI_ROOT"
if [[ -x "$OPENPI_ROOT/.venv/bin/python" ]]; then
  python_cmd=("$OPENPI_ROOT/.venv/bin/python")
else
  # Do not let verification mutate a working environment by syncing uv.lock.
  python_cmd=(uv run --no-sync python)
fi

"${python_cmd[@]}" - <<'PY'
import inspect

from openpi.models import pi0
from openpi.models import rtc
from openpi.policies import nero_bimanual_policy
from openpi.policies import nero_policy
from openpi.policies import policy
from openpi.training import config
from openpi.training import data_loader

data_fields = config.DataConfig.__dataclass_fields__
train_fields = config.TrainConfig.__dataclass_fields__
assert "lerobot_split_manifest" in data_fields
assert "lerobot_split" in data_fields
assert "gradient_accumulation_steps" in train_fields
assert hasattr(data_loader, "repair_lerobot_episode_subset_index")
assert hasattr(rtc, "guided_velocity")
assert hasattr(nero_policy, "NeroInputs")
assert hasattr(nero_bimanual_policy, "NeroBimanualInputs")

policy_parameters = inspect.signature(policy.Policy.infer).parameters
sample_parameters = inspect.signature(pi0.Pi0.sample_actions).parameters
for name in ("prev_chunk_left_over", "inference_delay", "execution_horizon"):
    assert name in policy_parameters, name
    assert name in sample_parameters, name

print("NERO_OPENPI_CORE_VERIFICATION_PASSED")
print("features=data_adapter,rtc,episode_split,gradient_accumulation")
PY

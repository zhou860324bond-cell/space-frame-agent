r"""端到端跑一遍 JSON 模型：校验 -> 求解 -> 自校验。

这就是 Agent 层将来要调的整条链路，只是把「LLM 产出 JSON」换成了「从文件读 JSON」。

    python examples\run_json.py examples\portal_frame_cases.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from frame3d import check_equilibrium, solve  # noqa: E402
from model_io import from_dict, validate_payload  # noqa: E402


from console import use_utf8  # noqa: E402  见 src/console.py：别让字符打死成功运行

use_utf8()
path = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/portal_frame.json")
payload = json.loads(path.read_text(encoding="utf-8"))

errors = validate_payload(payload)
if errors:
    print("校验未通过，这份清单原样回喂给模型即可重试：")
    for e in errors:
        print("  ", e)
    raise SystemExit(1)

model = from_dict(payload)
print(f"校验通过：{path.name}")
print(f"节点 {len(model.nodes)} 个，杆件 {len(model.members)} 根，"
      f"工况 {len(model.load_cases)} 个，组合 {len(model.combos)} 个")

sol = solve(model)

print(f"\n{'工况/组合':<22}{'最大合位移':>14}{'出现节点':>10}{'平衡校核':>12}")
print("-" * 60)
for name, result in sol.all_results().items():
    best_node, best = None, -1.0
    for nid in model.order():
        d = model.node_dofs(nid)
        mag = float(np.linalg.norm(result.U[d[:3]]))
        if mag > best:
            best, best_node = mag, nid
    eq = check_equilibrium(model, sol, name)
    flag = "通过" if eq["ok"] else f"残差 {eq['relative']:.1e}"
    print(f"{name:<22}{best * 1000:>12.4f} mm{best_node:>10}{flag:>12}")

controlling = max(sol.all_results().values(),
                  key=lambda r: max(abs(f).max() for f in r.member_forces.values()))
print(f"\n内力控制工况：{controlling.name}")
for mid in sorted(model.members):
    f = controlling.member_forces[mid]
    print(f"  杆件 {mid}: N={f[0] / 1e3:9.3f} kN  "
          f"Mz(i)={f[5] / 1e3:9.3f} kN·m  Mz(j)={f[11] / 1e3:9.3f} kN·m")

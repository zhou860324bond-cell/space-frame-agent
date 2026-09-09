"""Beta 0.1 五分钟主链路：一根物理构件，跨中荷载自动剖分并回聚结果。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import Session  # noqa: E402
from console import use_utf8  # noqa: E402


MODEL = {
    "schema_version": 1,
    "units": "N-m-Pa",
    "materials": [{"name": "Steel", "E": 2.0e11, "nu": 0.3}],
    "sections": [{"name": "S", "A": 0.01, "Iy": 1.0e-5,
                  "Iz": 2.0e-5, "J": 5.0e-6}],
    "nodes": [{"id": 11, "x": 0.0, "y": 0.0, "z": 0.0},
              {"id": 22, "x": 3.0, "y": 0.0, "z": 0.0}],
    "members": [{"id": 101, "i": 11, "j": 22,
                 "section": "S", "material": "Steel"}],
    "supports": [{"node": 11, "fix": [1, 1, 1, 1, 1, 1]}],
    "member_spans": [{"member": 101, "kind": "point",
                      "w1": [0.0, 0.0, -1000.0], "a": 1.5}],
}


def run() -> Session:
    session = Session()
    loaded = session.set_model(MODEL)
    assert loaded.ok, loaded.payload

    mesh = session.preview_analysis_mesh()
    assert mesh.ok, mesh.payload
    assert mesh.payload["physical_members"] == 1
    assert mesh.payload["analysis_elements"] == 2
    assert mesh.payload["member_mapping"] == {"101": [101, 102]}

    solved = session.solve_model()
    assert solved.ok, solved.payload
    assert session.result_db is not None

    diagram = session.query_diagram(member=101, component="Mz")
    assert diagram.ok, diagram.payload
    plotted = session.plot_results(kind="moment")
    assert plotted.ok, plotted.payload

    print("Beta 0.1 主链路通过")
    print("物理构件 101 -> 分析单元 101, 102")
    print(f"自动切分节点：{mesh.payload['split_node_details']}")
    print(f"Mz 峰值：{diagram.payload['peak']:.6g} {diagram.payload['unit']} "
          f"@ x={diagram.payload['at_x_m']:.3g} m")
    print(f"Result DB 模型哈希：{session.result_db.metadata['model_hash']}")
    print(f"结果图：{plotted.payload['path']}")
    return session


if __name__ == "__main__":
    use_utf8()
    run()

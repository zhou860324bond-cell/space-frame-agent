"""对单张图片跑一次多模态识别，并与真值逐项比对。

放在项目里任意位置都能跑（会向上找到含 src/ 的仓库根目录）。不带参数时
默认用同目录下的 frame_elevation.png / frame_elevation.truth.json：

    .venv\\Scripts\\python.exe tools\\recognize_one.py
    .venv\\Scripts\\python.exe tools\\recognize_one.py 图片.png 真值.json

密钥自动从仓库根目录的 deepseek.key 读（也认 DEEPSEEK_API_KEY 环境变量）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    """向上找含 src/frame3d.py 的目录。脚本被放到子目录里也能跑——
    上一版写死了 __file__ 的父目录当根，放错位置就是 ModuleNotFoundError。"""
    for candidate in (HERE, *HERE.parents):
        if (candidate / "src" / "frame3d.py").is_file():
            return candidate
    raise SystemExit(
        f"没找到项目根目录（含 src/frame3d.py）。当前脚本在 {HERE}，"
        "请把它放进项目文件夹内的任意位置再运行。")


ROOT = _repo_root()
for extra in ("src", "multimodal_eval"):
    sys.path.insert(0, str(ROOT / extra))

from multimodal_contract import file_digest                 # noqa: E402
from multimodal_workflow import MultimodalControllerState   # noqa: E402
from sketch_parser import SketchParser                      # noqa: E402
import credentials                                          # noqa: E402


def _resolve(argv: list[str]) -> tuple[Path, Path]:
    if len(argv) >= 3:
        return Path(argv[1]), Path(argv[2])
    image = HERE / "frame_elevation.png"
    truth = HERE / "frame_elevation.truth.json"
    if not image.is_file():
        raise SystemExit(
            f"没找到默认图片 {image}。请把图片和真值放在脚本同目录，"
            "或显式传两个参数：脚本 图片.png 真值.json")
    return image, truth


def main() -> int:
    image, truth_path = _resolve(sys.argv)
    if not image.is_file():
        raise SystemExit(f"图片不存在：{image}")
    if not truth_path.is_file():
        raise SystemExit(f"真值不存在：{truth_path}")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    key = credentials.load_api_key(ROOT)
    print("项目根目录:", ROOT)
    print("密钥来源  :", credentials.source(ROOT))
    print("密钥指纹  :", credentials.fingerprint(key))
    print("模型      :", SketchParser._default_model("deepseek"))
    print("图片      :", image.name)
    if not key:
        raise SystemExit("没读到密钥。请确认仓库根目录有 deepseek.key，"
                         "或设置环境变量 DEEPSEEK_API_KEY。")
    try:
        import openai  # noqa: F401
    except ImportError:
        raise SystemExit(
            "缺少 openai 依赖。先跑：\n"
            f'  "{ROOT}\\.venv\\Scripts\\python.exe" -m pip install '
            "-r requirements-multimodal.txt")

    state = MultimodalControllerState()
    state.load_image(file_digest(image))
    job = state.start_recognition("verify-deepseek")
    parsed = SketchParser.from_env("deepseek").parse_v2_with_retry(
        image, state, job, max_repairs=1)

    if not parsed.success:
        print("\n识别失败：")
        for err in parsed.errors:
            print("  -", err)
        print("\n若是 Connection error：这台机器到 api.deepseek.com 不通。")
        print("若是 model not found：模型名对不上，检查 sketch_parser 的默认值。")
        return 1

    draft = parsed.draft or {}
    got = draft.get("image_model") or {}
    print("\n%-10s %8s %8s" % ("项目", "真值", "识别"))
    for key_name in ("nodes", "members", "supports", "loads"):
        print("%-10s %8d %8d" % (key_name, len(truth.get(key_name, [])),
                                 len(got.get(key_name, []) or [])))
    print("\n尺度状态: 真值 %s / 识别 %s"
          % (truth.get("scale", {}).get("status"),
             (draft.get("scale") or {}).get("status")))
    issues = draft.get("issues") or []
    print("模型自己提出的疑问 %d 条：" % len(issues))
    for item in issues[:8]:
        print("  -", item if isinstance(item, str)
              else json.dumps(item, ensure_ascii=False)[:120])

    out = truth_path.with_suffix(".response.json")
    out.write_text(json.dumps(draft, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n完整草稿已写入", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""打包扁平 src 模块与 desktop 包。"""

from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent
setup(
    packages=["desktop"],
    package_dir={"": "src", "desktop": "desktop"},
    py_modules=sorted(path.stem for path in (ROOT / "src").glob("*.py")
                      if path.name != "__init__.py"),
)

from pathlib import Path
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, extension):
        source_dir = Path(__file__).parent.resolve()
        build_dir = Path(self.build_temp) / extension.name.replace(".", "_")
        output_dir = Path(self.get_ext_fullpath(extension.name)).parent.resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        subprocess.check_call([
            "cmake",
            "-S", str(source_dir),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DPython3_EXECUTABLE={__import__('sys').executable}",
        ])
        subprocess.check_call([
            "cmake", "--build", str(build_dir),
            "--config", "Release", "--target", "_wb_ai", "--parallel",
        ])


setup(
    name="wb-ai",
    version="0.2.0",
    description="A Python-callable local C++ language model library",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[CMakeExtension("wb_ai._wb_ai")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

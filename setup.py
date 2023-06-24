from skbuild import setup

setup(
    python_requires = ">=3.11",
    packages = ["racplusplus"],
    package_dir={"": "src"},
    cmake_install_dir="src/racplusplus"
)

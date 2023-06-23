from skbuild import setup

setup(
    name = "racplusplus",
    version = "0.0.1",
    description = "Reciprocal Agglomerative Clustering, optimized for speed/ parallelization in C++.",
    author = "Porter Hunley, Daniel Frees",
    author_email = "porterhunley@themedicalboard.net, danielfrees@g.ucla.edu",
    packages = ["racplusplus"],
    package_dir={"": "src"},
    cmake_install_dir="src/racplusplus",
    license = "MIT", 
    python_requires = ">=3.11"
)

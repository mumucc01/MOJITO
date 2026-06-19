import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name="diffusion_planner",
    version="1.0.0",
    author="Zheng Yinan, Ruiming Liang, Kexin Zheng @ Tsinghua AIR",
    packages=["diffusion_planner"],
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="MIT",
)

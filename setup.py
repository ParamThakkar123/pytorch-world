from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

README = (
    (HERE / "README.md").read_text(encoding="utf-8")
    if (HERE / "README.md").exists()
    else ""
)

setup(
    name="torchwm",
    version="0.1.0",
    description="A Pytorch Based library for training world models",
    long_description=README,
    long_description_content_type="text/markdown",
    author="",
    license="MIT",
    packages=find_packages(exclude=("tests", "results", "envs", ".venv", "venv")),
    include_package_data=True,
    python_requires=">=3.13",
    install_requires=[
        "ale-py>=0.11.2",
        "gym>=0.26.2",
        "gymnasium>=1.2.2",
        "mlagents-envs>=0.28.0",
        "moviepy>=2.2.1",
        "opencv-python>=4.12.0.88",
        "plotly>=6.5.0",
        "pre-commit>=4.5.0",
        "pygame>=2.6.1",
        "tensorboard>=2.20.0",
        "tensorboardx>=2.6.4",
        "tqdm>=4.67.1",
        "torch>=1.13.0",
        "torchvision>=0.14.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="world-models pytorch",
    zip_safe=False,
)

"""
LEAD 包安装配置。

支持通过 pip install -e . 进行可编辑安装。
"""

from setuptools import setup, find_packages


def read_requirements():
    """
    从 requirements.txt 读取依赖列表。

    Returns:
        list[str]: 依赖包名列表。
    """
    with open("requirements.txt", "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="lead",
    version="0.1.0",
    description="LEAD: Entropy-adaptive soft/hard decoding for VLM reasoning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="LEAD Authors",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lead-eval=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

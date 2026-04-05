from setuptools import setup, find_packages

setup(
    name="srfl",
    version="1.0.0",
    author="Bishal Neupane",
    author_email="cosmobishal@gmail.com",
    description="Swarm Renormalization Field Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "matplotlib>=3.5",
        "scipy>=1.9",
    ],
    entry_points={
        "console_scripts": [
            "srfl-run=srfl.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

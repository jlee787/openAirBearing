from setuptools import setup, find_packages

setup(
    name="OpenAir",
    version="0.1.0",
    packages=find_packages(),
    python_requires='>=3.10',  # Add this line
    install_requires=[
        "dash",
        "numpy",
        "plotly",
        "scipy",
        "callbacks"
    ],
)
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openairbearing",
    version="0.1.0",
    description="A Python package for externally pressurized air bearing analysis.",
    author="Mikael Miettinen",
    author_email="mikael.miettinen@iki.fi",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aalto-Arotor/openAirBearing",  
     project_urls={
        "Online demo": "https://www.openairbearing.com",
    },
    entry_points={
        "console_scripts": [
            "openairbearing=openairbearing.app.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "openairbearing": [
            "assets/style.css",
            "assets/favicon.ico",
        ],
    },
    packages=find_packages(exclude=["examples", "tests"]),
    python_requires=">=3.10",
    install_requires=[
        "dash>=2.0.0,<3.0.0",
        "numpy>=1.21.0,<2.0.0",
        "plotly>=6.0.0,<7.0.0",
        "scipy>=1.7.0,<2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
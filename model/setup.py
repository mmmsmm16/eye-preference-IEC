from setuptools import setup, find_packages

setup(
    name="iec-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "diffusers==0.31.0",
        "transformers==4.47.0",
        "torch==2.1.1",
        "accelerate==1.2.0",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pytest",
        "pytest-asyncio",
        "httpx"
    ],
)

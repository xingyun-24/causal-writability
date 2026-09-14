from setuptools import setup, find_packages

setup(
    name="seriality-gap",
    version="1.0.0",
    author="Jorge Diaz Chao",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"sshv2": ["experiments/*/config.yaml"]},
)

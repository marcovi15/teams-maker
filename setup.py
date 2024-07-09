from setuptools import setup, find_packages
import os

# Read the requirements from the requirements.txt file
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.read().splitlines()

setup(
    name="meetup-manager",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)
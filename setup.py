from setuptools import setup, find_packages
import os

setup(name="MolAlbert", packages=find_packages(include=["utils", "tests"]), scripts=[os.path.join("scripts", filename) for filename in os.listdir("scripts")])

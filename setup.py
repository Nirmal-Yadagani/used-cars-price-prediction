from setuptools import find_packages,setup
from typing import List

def get_requirements(requirements_path:str)->List[str]:
    '''
    This function will return the list of requirements.
    '''
    HYPEN_E_DOT = '-e .'
    list_pkgs = []
    with open(requirements_path,"r") as f:
        for pkg in f.readlines():
            list_pkgs.append(pkg.rstrip())

        if HYPEN_E_DOT in list_pkgs:
            list_pkgs.remove(HYPEN_E_DOT)

    return list_pkgs

setup(
    name="Used car price prediction",
    version="0.0.1",
    author="Nirmal Yadagani",
    author_email="nirmalproffesionallife@gmail.com",
    maintainer="Nirmal Yadagani",
    maintainer_email="nirmalproffesionallife@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
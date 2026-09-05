#youll be able to install the application as a package through setup.py.
#as in you can deploy this in pypi and install it using pip install <package_name> and use it
# -e . is used to install the package in editable mode, which means that any changes made to the source code will be reflected immediately without needing to reinstall the package. This is useful during development when you want to test changes without having to reinstall the package each time.
# -e . also an indcation that setup.py is present in the current directory and should be used to install the package. and automatically setup thing will get built and installed in the current environment.
from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements mentioned in requirements.txt file
    '''
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='test_pckg',
    version='0.1',
    packages=find_packages(),
    author_email='niviiveera@gmail.com' ,
    author='Nivi Veera',
    install_requires=get_requirements('requirements.txt')
)
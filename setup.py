
from setuptools import setup, find_packages
import platform
import sys

kwargs = {'author': '',
 'author_email': '',
 'classifiers': ['Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering'],
 'description': '',
 'download_url': '',
 'include_package_data': True,
 'install_requires': ['oct2py'],
# 'keywords': ['openmdao'],
 'license': '',
 'maintainer': '',
 'maintainer_email': '',
 'name': 'becas_wrapper',
 'packages': ['becas_wrapper'],
 'dependency_links':['https://github.com/blink1073/oct2py/tarball/master#egg=oct2py'],
 'url': '',
 'version': '0.1',
 'zip_safe': False}

if 'Windows' not in platform.platform():
    kwargs['install_requires'].append('pexpect')


setup(**kwargs)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join as pjoin

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev0'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "locator : Find earthquake locations"
long_description = """
LOCATOR
=======

License
=======
``LOCATOR`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2018 - Simon Staehler (mail@simonstaehler.com)
"""

opts = dict(name='locator',
            maintainer='Simon Staehler',
            maintainer_email='mail@simonstaehler.com',
            description=description,
            long_description=long_description,
            url='http://github.com/sstaehler/locator',
            license='none',
            classifiers=CLASSIFIERS,
            author='Simon Staehler',
            author_email='mail@simonstaehler.com',
            platforms='OS Independent',
            version=__version__,
            packages=['locator'],
            package_data={'locator': [pjoin('data', '*')]},
            install_requires=['numpy', 'scipy', 'h5py', 'pyyaml', 'obspy'],
            entry_points={'console_scripts':
                          ['locator = locator.main:main', ], })


if __name__ == '__main__':
    setup(**opts)


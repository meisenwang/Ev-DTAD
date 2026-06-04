#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'psee_adt',
    version          = '0.0.1',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = "Packaged automotive dataset evaluation toolbox",
    packages         = setuptools.find_packages(
        include = [ 'psee_adt', 'psee_adt.*' ]
    ),
    install_requires = [
        'opencv-python',
        'numpy',
        'pycocotools',
    ],
)

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install


class move_ttf(install):
    def run(self):
        """
        Adopted from https://github.com/breedlun/clearplot/blob/master/setup.py

        Performs the usual install process and then copies the True Type fonts
        that come with clearplot into matplotlib's True Type font directory,
        and deletes the matplotlib fontList.cache
        """
        # Perform the usual install process
        install.run(self)
        # Try to install custom fonts
        # try:
        import os
        import shutil
        import dohlee as doh
        import matplotlib as mpl
        import matplotlib.font_manager as font_manager

        # Find where matplotlib stores its True Type fonts
        mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
        mpl_ttf_dir = os.path.join(mpl_data_dir, 'fonts', 'ttf')
        if not os.path.exists(mpl_ttf_dir):
            os.makedirs(mpl_ttf_dir)

        # Copy the font files to matplotlib's True Type font directory
        # (I originally tried to move the font files instead of copy them,
        # but it did not seem to work, so I gave up.)
        doh_ttf_dir = os.path.join(os.path.dirname(doh.__file__), 'fonts')
        for file_name in os.listdir(doh_ttf_dir):
            if file_name[-4:] == '.ttf':
                old_path = os.path.join(doh_ttf_dir, file_name)
                new_path = os.path.join(mpl_ttf_dir, file_name)
                shutil.copyfile(old_path, new_path)
                print("Copying " + old_path + " -> " + new_path)

        # Try to delete matplotlib's fontList cache
        mpl_cache_dir = mpl.get_cachedir()
        mpl_cache_dir_ls = os.listdir(mpl_cache_dir)
        font_list_cache_names = ["fontList.cache", "fontList.py3k.cache"]
        for font_list_cache_name in font_list_cache_names:
            if font_list_cache_name in mpl_cache_dir_ls:
                fontList_path = os.path.join(mpl_cache_dir, font_list_cache_name)
                os.remove(fontList_path)
                print("Deleted the matplotlib " + font_list_cache_name)

        # except Exception as e:
        #     print(Exception)
        #     warnings.warn("WARNING: An issue occured while installing the custom fonts for dohlee.plot.")


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


setup(
    name='dohlee',
    version='0.1.16',
    license='MIT license',
    description='My personal python utility library.',
    # long_description='%s\n%s' % (
    #     re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
    #     re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    # ),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author='Dohoon Lee',
    author_email='apap950419@gmail.com',
    url='https://github.com/dohlee/python-dohlee',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
        'matplotlib>=3.0.0',
        'numpy>=1.14.2',
        'pandas>=0.23.4',
        'scikit-learn>=0.19.1',
        'scipy>=1.0.1',
        'seaborn>=0.8.1',
        'six>=1.11.0',
        'sklearn>=0.0',
        'mygene>=3.0.0',
        'pyensembl>=1.2.6',
        'tqdm>=4.23.0',
        'adjustText>=0.7.3',
        # 'fastTSNE>=0.2.6',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        'console_scripts': [
            'dohlee = dohlee.cli:main',
        ]
    },
    package_data={
        '': ['fonts/*.ttf']
    },
    cmdclass={
        'install': move_ttf
    },
)

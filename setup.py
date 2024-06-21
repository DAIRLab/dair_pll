from setuptools import setup
from pip._internal import main as pipmain

install_reqs = [
    # library
    'torch',
    'moviepy',
    'Pillow',
    'wandb',
    'mujoco-py',
    'optuna',
    'numpy==1.26.4',
    'scipy',
    'typing_extensions',
    'matplotlib',
    'threadpoolctl',
    'click',
    'pywavefront',
    'python-fcl',
    # documentation
    'networkx',
    'pydeps',
    'Sphinx',
    'sphinx-autodoc-typehints',
    'sphinx-rtd-theme',
    'sphinx-toolbox',
    'sphinxcontrib-bibtex',
    # development
    'yapf',
    'pylint',
    'mypy',
    # git
    'drake-pytorch @ git+https://github.com/DAIRLab/drake-pytorch.git#egg=drake-pytorch-0.1',
    'sappy @ git+https://github.com/mshalm/sappy.git#egg=sappy-0.0.1',
]

try:
    # Note: pydrake needs numpy, install numpy manually first
    try:
        import numpy
    except ModuleNotFoundError as e:
        pipmain(['install', 'numpy'])
    import pydrake
    print('USING FOUND DRAKE VERSION')
except ModuleNotFoundError as e:
    install_reqs += ['drake']

setup(
    name='dair_pll',
    version='0.0.1',
    packages=['dair_pll'],
    install_requires=install_reqs,
)

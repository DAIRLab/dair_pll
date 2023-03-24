from setuptools import setup

install_reqs = [
    # library
    'torch',
    'moviepy',
    'Pillow',
    'wandb',
    'mujoco-py',
    'optuna',
    'numpy',
    'scipy',
    'typing_extensions',
    'matplotlib',
    'threadpoolctl',
    'click',
    'pywavefront',
    'python-fcl',
    # documentation
    'networkx',
    'pydeps==1.10.12',
    'Sphinx',
    'sphinx-autodoc-typehints',
    'sphinx-rtd-theme',
    'sphinx-toolbox',
    'sphinxcontrib-bibtex',
    # development
    'yapf',
    'pylint',
    'mypy',
]

try:
    import pydrake

    print('USING FOUND DRAKE VERSION')
except ModuleNotFoundError as e:
    install_reqs += ['drake']

dependency_links = [
    'git+https://github.com/DAIRLab/drake-pytorch.git',
    'git+https://github.com/mshalm/sappy.git'
]

setup(
    name='dair_pll',
    version='0.0.1',
    packages=['dair_pll'],
    install_requires=install_reqs,
    dependency_links=dependency_links
)

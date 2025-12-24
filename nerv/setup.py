from setuptools import find_packages, setup

install_requires = [
    'numpy', 'pyyaml', 'six', 'tqdm', 'opencv-python', 'matplotlib', 'open3d',
    'wandb', 'moviepy', 'imageio', 'torch', 'torchvision', 'torchmetrics',
    'pytorch-lightning'
]


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'nerv/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name='nerv',
    version=get_version(),
    author='Ziyi Wu',
    author_email='dazitu616@gmail.com',
    license='MIT',
    description='Ziyi Wu\'s personal research python toolbox',
    long_description=readme(),
    keywords='computer vision',
    packages=find_packages(),
    url='https://github.com/Wuziyi616/nerv',
    install_requires=install_requires,
)  # yapf: disable

from distutils.core import setup
from setuptools import find_packages

setup(
    name='MAPLE',
    packages=find_packages(),
    version='0.0.1',
    description='Offline Model-based Adaptable Policy Learning',
    long_description=open('./README.md').read(),
    author='Xiong-Hui Chen, Fan-Ming Luo',
    author_email='chenxh@lamda.nju.edu.cn, luofm@lamda.nju.edu.cn',
    entry_points={
        'console_scripts': (
            'mopo=softlearning.scripts.console_scripts:main',
            'viskit=mopo.scripts.console_scripts:main'
        )
    },
    install_requires=[
        "RLA @ git+https://github.com/polixir/RLAssistant.git@main#egg=RLA",
        "serializable @ git+https://github.com/hartikainen/serializable.git@76516385a3a716ed4a2a9ad877e2d5cbcf18d4e6#egg=serializable",
        'gtimer',
        'dotmap',
    ],
    zip_safe=True,
    license='MIT'
)

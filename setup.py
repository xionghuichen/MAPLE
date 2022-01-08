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
        "RLA>=0.3",
    ],
    zip_safe=True,
    license='MIT'
)

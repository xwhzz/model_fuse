from setuptools import setup, find_packages

setup(
    name='opt',
    version='0.1',
    packages=find_packages(),
    description='Fuse model',
    author='xwhzz',
    url='https://github.com/yourusername/mypackage',
    install_requires=[
        'onnx'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',  # Minimum version requirement of the package
)

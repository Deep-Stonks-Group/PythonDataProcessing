from setuptools import find_packages, setup

setup(
    name = 'PythonDataProcessing',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
         'numpy==1.19.4',
         'torch==1.8.1',
         'torchvision==0.9.1',
         'Historic-Crypto==0.1.4',
         'yfinance==0.1.59',
         'TA-Lib==0.4.20',
         'sklearn==0.0',
         'matplotlib==3.3.3'
    ]
)

from setuptools import find_packages, setup

setup(
    name = 'PythonDataProcessing',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
         'numpy==1.19.4',
         'Historic-Crypto==0.1.4',
         'yfinance==0.1.63',
         'sklearn==0.0',
         'TA-Lib==0.4.20',
    ]
)

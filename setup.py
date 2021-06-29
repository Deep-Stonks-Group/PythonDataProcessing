from setuptools import find_packages, setup

setup(
    name = 'PythonDataProcessing',
    packages = find_packages(),
    include_package_data = True,
    setup(
    name = 'forecast-team-1',
    packages = find_packages(),
    include_package_data = True,
    dependency_links=[
        "git+git://github.abc.com/abc/SomePrivateLib.git#egg=SomePrivateLib",
    ]
)

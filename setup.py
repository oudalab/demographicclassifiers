from setuptools import setup, find_packages
setup(
    name='democlassifiers',
    version='0.1',
    author='',
    authour_email='cgrant@ou.edu',
    packages=find_packages(exclude=('tests', 'docs')),
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)

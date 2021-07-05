from setuptools import find_packages, setup

setup(
    name='qdms',
    packages=find_packages(include=['qdms']),
    version='0.1',
    description='This is a description.',
    author='Sébastien Graveline',
    author_email='Sebastien.Graveline@usherbrooke.ca',
    license='NotYet',
    url='https://github.com/Talgarr/TestOfficial/',
    install_requires=['numpy', 'matplotlib', 'h5py'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==6.2.4'],
    test_suite='tests',
)

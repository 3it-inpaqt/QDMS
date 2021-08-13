from setuptools import find_packages, setup

setup(
    name='qdms',
    packages=find_packages(include=['qdms']),
    version='1.0',
    description='This is a description.',
    author='Sébastien Graveline',
    author_email='Sebastien.Graveline@usherbrooke.ca',
    license='MIT',
    url='https://github.com/3it-nano/QDMS/',
    install_requires=['numpy', 'matplotlib', 'cpickle'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==6.2.4'],
    test_suite='tests',
)

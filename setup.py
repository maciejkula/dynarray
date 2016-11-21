from setuptools import setup

setup(
    name='dynarray',
    version='0.1.0',
    description='Dynamically growable numpy arrays.',
    url='https://github.com/maciejkula/dynarray',
    download_url='https://github.com/maciejkula/dynarray/tarball/0.1.0',
    packages=['dynarray'],
    install_requires=['numpy'],
    tests_require=['pytest', 'hypothesis[numpy]'],
    author='Maciej Kula',
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License']
)

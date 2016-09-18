from setuptools import setup

setup(
    name='dynarray',
    version='0.1.0',
    description='LightFM recommendation model',
    url='https://github.com/lyst/lightfm',
    download_url='https://github.com/lyst/lightfm/tarball/1.9',
    packages=['dynarray'],
    install_requires=['numpy'],
    tests_require=['pytest', 'hypothesis[numpy]'],
    author='Maciej Kula',
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)

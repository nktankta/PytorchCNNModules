from setuptools import setup

package_name = "PytorchCNNModules"
requires = ['torch >= 1.4']


setup(
    name='PytorchCNNModules',
    version='0.0.1',
    packages=['PytorchCNNModules'],
    url='https://github.com/nktankta/PytorchCNNModules',
    license='MIT License',
    author='nktankta',
    author_email='nakataatsuya@gmai.com',
    description='Pytorch CNN Module collection',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Natural Language :: Japanese',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=requires,
    keywords='pytorch cnn module inception block',
)

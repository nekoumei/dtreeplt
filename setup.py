from setuptools import setup, find_packages
from os import path
import re

root_dir = path.abspath(path.dirname(__file__))
package_name = 'dtreeplt'

try:
    with open('README.md') as f:
        readme = f.read()
except IOError:
    readme = ''


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

assert version

setup(
    name=package_name,
    version=version,
    url='https://github.com/nekoumei/dtreeplt',
    author='nekoumei',
    author_email='nekoumei@gmail.com',
    maintainer='nekoumei',
    maintainer_email='nekoumei@gmail.com',
    description='Visualize Decision Tree without Graphviz.',
    long_description=readme,
    packages=find_packages(),
    install_requires=_requires_from_file('requirements.txt'),
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Environment :: MacOS X',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
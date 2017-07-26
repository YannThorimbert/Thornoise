from setuptools import find_packages, setup
import thornoise

setup(name='thornoise',
      version=thornoise.__version__,
      description='Library for making noise in Python2 and Python3',
      author='Yann Thorimbert',
      author_email='yann.thorimbert@gmail.com',
      url='http://www.thorpy.org/',
      keywords=['pygame','noise','terrain','generation'],
      packages=find_packages(),
      include_package_data=True,
      license='MIT')

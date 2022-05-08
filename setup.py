from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='Sat',
  version='0.0.1',
  description='Analysis of satellite images',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='',
  author_email='',
  license='MIT', 
  classifiers=classifiers,
  keywords='analysis', 
  packages=find_packages(),
  install_requires=[
    "<matplotlib>;python_version<'<3.6.6>'",
    "<numpy> >= <1.17.4>",
    "<rasterio> >= <1.2.10>",
    "<plotly> >= <5.7.0>",
    "<earthpy>",
    "<tensorboard>",
   ] 

)
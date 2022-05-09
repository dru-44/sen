from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='Satt2',
  version='0.0.1',
  description='Sentinel satellite image module',
  long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.md').read(),
  url='https://github.com/dru-44/sen.git',  
  author='Dru44',
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
    "<rich>",
    "<lightgbm>",
    "<seaborn>",
   ] 

)
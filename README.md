
# Sentinel 2 image package

Under construction! Not ready for use yet! Currently experimenting and planning!

## Setup
```python
pip install Sentinel-imgpackage
```

## Example of How To Use (Alpha Version)

Creating obj

```python

from senpkg import *

path="your data source path"
gt="ground-truth .mat name"
mname="model name"

p= Analysis.senA(path)
q=LandCoverCla.senC(path,gt)
r=LandCoverMod.senM(path,gt,mname)

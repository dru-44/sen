
# Sentinel 2 image package

Under construction! Not ready for use yet! Currently experimenting and planning!


## Example of How To Use (Alpha Version)

Creating obj

```python

from senpkg import *

path="your data source path"
p= Analysis.senA(path)
q=LandCoverCla.senC(path)
r=LandCoverMod.senM(path)

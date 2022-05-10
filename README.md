<div id="top"></div>

# Sentinel 2 image package 
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 

Under construction! Not ready for use yet! Currently experimenting and planning!


![forthebadge made-with-python](https://img.shields.io/pypi/v/Sentinel-imgpackage?style=for-the-badge)
<!-- GETTING STARTED -->
## Getting Started

This is an example of how you can set up your project.

Follow these simple example steps.

### Prerequisites

This project requires ``python3`` installed.


### Installation



1. Clone the dataset repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install pip packages
   ```python
   pip install Sentinel-imgpackage
   ```

## Usage

Model implementation requires sufficient amount of GPU and CPU resources.

Best when used in virtual notebooks like
* Google Colab Pro 
* Gradient 



## Example of How To Use (Alpha Version)

Creating obj

```python

from senpkg import *

path="your data source path"
gt="ground-truth .mat filename"
mname="model name"

p= Analysis.senA(path)
q=LandCoverCla.senC(path,gt)
r=LandCoverMod.senM(path,gt,mname)
```
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

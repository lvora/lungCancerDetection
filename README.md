![Logo of the project](examples/1_0015ceb851d7251b8f399e39779d1e7d.gif)

# Kaggle Data Science Bowl 2017 - ECE 6254
> Class project submission


## Resources
<https://www.kaggle.com/c/data-science-bowl-2017>

<https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial>


## Installing / Getting started

A minimal setup you need to get running.

### Ubuntu 14.04
#### Get necessary packages
```shell
sudo apt-get update
sudo apt-get install git python3 python3-pip
sudo -H pip3 install virutualenvwrapper
```

#### Set up the environment
```shell
mkdir ~/projects && cd ~/projects
git clone https://github.gatech.edu/kds17/kds17.git
mkvirtualenv kds17
echo "proj_name=\$(basename \$VIRTUAL_ENV) 
cd ~/projects/\$proj_name" >> ~/.venv/postactivate
deactivate
workon kds17
pip install -r requirements.txt
python
```

If you do not have CUDA and cuDNN installed then pip install will fail with tensorflow-gpu.  Just install the regular one in your virtual environment:

```shell
pip install tensorflow
```

#### Test the environment
```python
import cv2
import tensorflow as tf
```
If you did not receive any errors while importing the packages then you should be good to go!

## Licensing

The code in this project is licensed under MIT license.

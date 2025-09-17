# To start

Here is the test using the stream from Jake's paper.

The basic work flow is stream fitting, then

## Stream visualization

The stream is 3D. We use only 2D (x-z), for plotting we name it as (x-y)

```{code-block} python

>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> name='fig5_streamB.npy'
>>> file_dir='enter_your_file_dir' # enter your own file_dir
>>> data=np.load(file_dir+name,allow_pickle='True')

>>> x=data.item()['x']
>>> y=data.item()['z']

>>> plt.figure(figsize=(5,5))
>>> plt.plot(x,y,'.')
```

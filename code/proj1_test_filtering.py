from gauss2D import gauss2D
from my_imfilter import my_imfilter
from normalize import normalize
from skimage.transform import resize
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

''' set up '''
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = main_path + '/data/cat.bmp'
test_image = mpimg.imread(img_path)
test_image = resize(test_image, (np.array(test_image.shape[:2])*0.7).astype(int))
test_image = test_image.astype(np.single)/255
plt.figure('Image')
plt.imshow(test_image)

# ... rest of your code ...

plt.show()
print('done')

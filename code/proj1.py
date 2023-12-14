from my_imfilter import my_imfilter
from vis_hybrid_image import vis_hybrid_image
from normalize import normalize
from gauss2D import gauss2D
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import os
from skimage.transform import resize

''' Setup '''
# read images and convert to floating point format
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image1 = mpimg.imread(main_path + '/data/bird.bmp')
image2 = mpimg.imread(main_path + '/data/plane.bmp')

# Resize images if they are not of the same size
if image1.shape != image2.shape:
    image1 = resize(image1, image2.shape)

image1 = image1.astype(np.single)/255 
image2 = image2.astype(np.single)/255

''' Filtering and Hybrid Image construction '''
cutoff_frequency = 7
gaussian_filter = gauss2D(shape=(cutoff_frequency*4+1,cutoff_frequency*4+1), sigma = cutoff_frequency)

low_frequencies = my_imfilter(image1, gaussian_filter)
high_frequencies = image2 - my_imfilter(image2, gaussian_filter)
hybrid_image = low_frequencies+high_frequencies

''' Visualize and save outputs '''
plt.figure(1)
plt.imshow(normalize(low_frequencies))
plt.figure(2)
plt.imshow(normalize(high_frequencies+0.5))
vis = vis_hybrid_image(hybrid_image)
plt.figure(3)
plt.imshow(normalize(vis))
plt.imsave(main_path+'/results/low_frequencies_mouse_bear.png', normalize(low_frequencies), 'quality', 95)
plt.imsave(main_path+'/results/high_frequencies_mouse_bear.png', normalize(high_frequencies+0.5), 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_mouse_bear.png', normalize(hybrid_image), 'quality', 95)
plt.imsave(main_path+'/results/hybrid_image_scales_mouse_bear.png', normalize(vis), 'quality', 95)
plt.show()

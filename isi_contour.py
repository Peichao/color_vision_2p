import skimage
from skimage import feature
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Ocular Dominance
file_path = "G:\\AE5\\Right Hemi\\OD.tif"
isi_map = Image.open(file_path)
isi_map = np.array(isi_map)

isi_map_od = (isi_map - isi_map.min()) / (isi_map.max() - isi_map.min())

edges_od = feature.canny(isi_map, sigma=5)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax1.imshow(isi_map_od, interpolation='nearest', cmap='gray')
ax2.imshow(edges_od, cmap='gray')

ax1.axis('off')
ax2.axis('off')

plt.show()

# RG Color Patches
file_path = "G:\\AE5\\Right Hemi\\RGLum.tif"
isi_map = Image.open(file_path)
isi_map = np.array(isi_map)

isi_map_col = (isi_map - isi_map.min()) / (isi_map.max() - isi_map.min())

edges_col = feature.canny(isi_map, sigma=3)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax1.imshow(isi_map_col, interpolation='nearest', cmap='gray')
ax2.imshow(edges_col, cmap='gray')

ax1.axis('off')
ax2.axis('off')

plt.show()

fig, ax = plt.subplots()
ax.imshow(edges_od, interpolation='nearest', cmap='jet', alpha=0.5)
ax.imshow(edges_col, cmap='gray', alpha=0.5)

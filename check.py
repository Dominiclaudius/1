from PIL import Image
import numpy as np

mask = Image.open(r'E:\mobile\mobile-deeplab-v3-plus-master\datasets\VOCdevkit\VOC2012\SegmentationClassRaw\2007_000032.png')
mask_np = np.array(mask)

print('Shape:', mask_np.shape)
print('Unique values:', np.unique(mask_np))
print('Dtype:', mask_np.dtype)


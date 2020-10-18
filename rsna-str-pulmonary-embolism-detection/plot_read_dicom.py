import matplotlib.pyplot as plt
import pydicom

print(__doc__)

dataset = pydicom.dcmread("data/train/0a4d7c9fa082/38eddda9207f/d8f1fda6f89a.dcm")

print("StudyInstanceUID.....:", dataset.StudyInstanceUID)
print("SeriesInstanceUID....:", dataset.SeriesInstanceUID)
print("SOPInstanceUID.......:", dataset.SOPInstanceUID)

print("Dataset.......:", dataset)

if 'PixelData' in dataset:
    rows = int(dataset.Rows)
    cols = int(dataset.Columns)
    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(dataset.PixelData)))
    if 'PixelSpacing' in dataset:
        print("Pixel spacing....:", dataset.PixelSpacing)

plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
plt.show()

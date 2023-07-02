import numpy as np
import cv2
from osgeo import gdal
import scipy.io

class DataPreparer:
    @staticmethod
    def clip_img(output, inp_file, prj, xRes, yRes, extent, nodata=None):
        outDs = gdal.Warp(output,
                          inp_file,
                          dstSRS=prj,
                          xRes=xRes,
                          yRes=yRes,
                          outputBounds=extent,
                          srcNodata=nodata,
                          resampleAlg='cubic',
                          format='GTiff')

        outDs = None

    @staticmethod
    def get_bounds(ds):
        xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
        width, height = ds.RasterXSize, ds.RasterYSize
        xmax = xmin + width * xpixel
        ymin = ymax + height * ypixel
        return xmin, ymin, xmax, ymax

    @staticmethod
    def create_polygons(img, xsize, ysize, px_size):
        xsize *= px_size
        ysize *= px_size

        xmin, ymin, xmax, ymax = DataPreparer.get_bounds(img)

        n_x = (xmax - xmin) // xsize
        x_overlap = (((xmax - xmin) % xsize) / n_x) * px_size

        n_y = (ymax - ymin) // ysize
        y_overlap = (((ymax - ymin) % ysize) / n_y) * px_size

        stepx = xsize - x_overlap
        stepy = ysize - y_overlap

        x = xmin
        y = ymin

        polygons = []

        x_ar = np.arange(xmin, xmax, stepx)
        y_ar = np.arange(ymin, ymax, stepy)

        for i in range(len(x_ar)):
            for j in range(len(y_ar)):
                if x_ar[i] <= xmin:
                    x_0 = xmin
                    x_1 = x_0 + xsize
                else:
                    x_0 = x_ar[i] - x_overlap
                    x_1 = x_0 + stepx + x_overlap

                if x_1 > xmax:
                    x_1 = xmax
                    x_0 = x_1 - xsize

                if y_ar[j] <= ymin:
                    y_0 = ymin
                    y_1 = y_0 + ysize
                else:
                    y_0 = y_ar[j] - y_overlap
                    y_1 = y_0 + stepy + y_overlap

                if y_1 > ymax:
                    y_1 = ymax
                    y_0 = y_1 - ysize

                polygons.append([x_0, y_0, x_1, y_1])

        return polygons

    @staticmethod
    def normalize(array, bands=[]):
        newarray = []

        if not bands:
            bands = [i for i in range(array.shape[0])]

        for b in bands:
            ar = array[b]
            max2 = list(ar.reshape(-1))
            max2.sort()
            max2 = max2[int(len(max2) * .99)]
            ar = (ar - ar.min()) / (max2 - ar.min())
            ar[ar > 1] = 1
            ar *= 255
            newarray.append(ar)

        return cv2.merge(np.array(newarray, int))

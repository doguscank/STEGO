import os
from tqdm import tqdm
from osgeo import gdal
import scipy.io
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from data_preparation import DataPreparer
import shutil
from omegaconf import DictConfig, OmegaConf
import hydra


class ImageProcessor:
    def __init__(self, input_path, output_path, size=224):
        self.input_path = input_path
        self.output_path = output_path
        self.size = size
        self.i_outpath = os.path.join(output_path, "imgs")
        self.outpath_tif = os.path.join(output_path, "original_chips")

    def process_images(self):
        if not os.path.exists(self.i_outpath):
            os.makedirs(self.i_outpath)
        else:
            shutil.rmtree(self.output_path)
            os.makedirs(self.i_outpath)

        if not os.path.exists(self.outpath_tif):
            os.makedirs(self.outpath_tif)

        for i in tqdm(os.listdir(self.input_path)):
            inp = os.path.join(self.input_path, i)

            img_inp = gdal.Open(inp)
            res = img_inp.GetGeoTransform()[1]
            proj = img_inp.GetProjection()
            polygons = DataPreparer.create_polygons(img_inp, self.size, self.size, res)
            for n, p in enumerate(tqdm(polygons), 1):
                output = os.path.join(self.i_outpath, f"{n}.mat")
                output_geo = os.path.join(self.outpath_tif, f"{n}.tif")
                if not os.path.exists(output):
                    DataPreparer.clip_img(output_geo, img_inp, proj, res, res, p)
                    arr = gdal.Open(output_geo).ReadAsArray()
                    nodata = arr[arr == 0]
                    if len(arr.shape) == 2:
                        arr = np.array([arr])
                    arr = DataPreparer.normalize(arr)
                    N = np.prod(arr.shape)
                    if len(nodata) == 0:
                        scipy.io.savemat(output, mdict={"img": arr})

        all_file = open(os.path.join(self.output_path, "all.txt"), mode="w")
        unlab_file = open(
            os.path.join(self.output_path, "unlabelled_train.txt"), mode="w"
        )
        for n in range(1, len(polygons) + 1):
            all_file.write(f"{n}\n")
            unlab_file.write(f"{n}\n")
        all_file.close()
        unlab_file.close()


class DatasetSplitter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.i_outpath = os.path.join(output_path, "imgs")

    def split_dataset(self):
        X = [i for i in os.listdir(self.i_outpath) if i.endswith(".mat")]
        x_train, x_val = train_test_split(X)

        train_path = os.path.join(self.i_outpath, "train")
        val_path = os.path.join(self.i_outpath, "val")

        if not os.path.exists(train_path):
            os.mkdir(train_path)

        for i in x_train:
            f = os.path.join(self.i_outpath, i)
            out = os.path.join(train_path, i)
            shutil.copyfile(f, out)

        if not os.path.exists(val_path):
            os.mkdir(val_path)

        for i in x_val:
            f = os.path.join(self.i_outpath, i)
            out = os.path.join(val_path, i)
            shutil.copyfile(f, out)

        train_file = os.path.join(self.i_outpath, "train")
        val_file = os.path.join(self.i_outpath, "val")

        with open(os.path.join(self.output_path, "unlabelled_train.txt"), "w") as f:
            for i in os.listdir(train_file):
                n = f"{os.path.splitext(i)[0]}\n"
                f.write(n)

        with open(os.path.join(self.output_path, "unlabelled_test.txt"), "w") as f:
            for i in os.listdir(val_file):
                n = f"{os.path.splitext(i)[0]}\n"
                f.write(n)


@hydra.main(config_path="configs", config_name="train_config.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_path = hydra.utils.to_absolute_path(cfg.pytorch_data_dir)
    output_path = hydra.utils.to_absolute_path(cfg.output_root)
    size = cfg.res

    # Image Processing
    image_processor = ImageProcessor(input_path, output_path, size)
    image_processor.process_images()

    # Dataset Splitting
    dataset_splitter = DatasetSplitter(output_path)
    dataset_splitter.split_dataset()


if __name__ == "__main__":
    my_app()

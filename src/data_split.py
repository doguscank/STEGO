import os
import shutil
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.i_outpath = os.path.join(output_path, 'imgs')

    def split_dataset(self):
        X = [i for i in os.listdir(self.i_outpath) if i.endswith('.mat')]
        x_train, x_val = train_test_split(X)

        train_path = os.path.join(self.i_outpath, 'train')
        val_path = os.path.join(self.i_outpath, 'val')

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

        train_file = os.path.join(self.i_outpath, 'train')
        val_file = os.path.join(self.i_outpath, 'val')

        with open(os.path.join(self.output_path, 'unlabelled_train.txt'), 'w') as f:
            for i in os.listdir(train_file):
                n = f'{os.path.splitext(i)[0]}\n'
                f.write(n)

        with open(os.path.join(self.output_path, 'unlabelled_test.txt'), 'w') as f:
            for i in os.listdir(val_file):
                n = f'{os.path.splitext(i)[0]}\n'
                f.write(n)

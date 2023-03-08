from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder

        # get the name of all files in the image_folder
        files = os.listdir(image_folder)
        # sort the files by name
        files.sort()
        # get the name of all images
        self.image_paths = [os.path.join(image_folder, file) for file in files]

        self.transform = transform
        
    def get_class_label(self, index):
        return 0
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        y = self.get_class_label(index)
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_paths)


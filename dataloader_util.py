from option import *
import os.path
import torch.utils.data as data
from PIL import Image
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def rotate(img, rotate_index):
    if rotate_index == 0:
        return img
    if rotate_index == 1:
        return img.rotate(90)
    if rotate_index == 2:
        return img.rotate(180)
    if rotate_index == 3:
        return img.rotate(270)
    if rotate_index == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)


class TrainLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/input')
        self.dir_B = os.path.join(self.root, self.phase + '/GT')
        self.dir_C = os.path.join(self.root, self.phase + '/T')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))

        self.transform = ToTensor()

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index]).convert("RGB")

        resized_a = A.resize((280, 280), Image.LANCZOS)
        resized_b = B.resize((280, 280), Image.LANCZOS)
        resized_c = C.resize((280, 280), Image.LANCZOS)

        w, h = resized_a.size
        x, y = randrange(w - self.fineSize + 1), randrange(h - self.fineSize + 1)
        cropped_a = resized_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_b = resized_b.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_c = resized_c.crop((x, y, x + self.fineSize, y + self.fineSize))

        rotate_index = randrange(0, 8)
        rotated_a = rotate(cropped_a, rotate_index)
        rotated_b = rotate(cropped_b, rotate_index)
        rotated_c = rotate(cropped_c, rotate_index)

        tensor_a = self.transform(rotated_a)
        tensor_b = self.transform(rotated_b)
        tensor_c = self.transform(rotated_c)

        return tensor_a, tensor_b, tensor_c

    def __len__(self):
        return len(self.A_paths)


class TrainUnlabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/input')
        self.dir_D = os.path.join(self.root, self.phase + '/candidate')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.D_paths = sorted(make_dataset(self.dir_D))

        self.transform = ToTensor()

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")

        A = A.resize((self.fineSize, self.fineSize), Image.LANCZOS)

        strong_data = data_aug(A)
        tensor_w = self.transform(A)
        tensor_s = self.transform(strong_data)

        name = self.A_paths[index]

        return tensor_w, tensor_s, name

    def __len__(self):
        return len(self.A_paths)


class ValLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/input')
        self.dir_B = os.path.join(self.root, self.phase + '/GT')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.transform = ToTensor()

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        resized_a = A.resize((self.fineSize, self.fineSize), Image.LANCZOS)
        resized_b = B.resize((self.fineSize, self.fineSize), Image.LANCZOS)

        tensor_a = self.transform(resized_a)
        tensor_b = self.transform(resized_b)

        return tensor_a, tensor_b

    def __len__(self):
        return len(self.A_paths)


class TestData(data.Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.root = dataroot

        self.dir_A = os.path.join(str(self.root) + '/input')

        self.A_paths = sorted(make_dataset(self.dir_A))

        self.transform = ToTensor()

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")

        tensor_a = self.transform(A)

        return tensor_a

    def __len__(self):
        return len(self.A_paths)


def data_aug(images):
    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
    strong_aug = images
    if random.random() < 0.8:
        strong_aug = color_jitter(strong_aug)
    strong_aug = transforms.RandomGrayscale(p=0.2)(strong_aug)
    if random.random() < 0.5:
        strong_aug = blurring_image(strong_aug)
    return strong_aug

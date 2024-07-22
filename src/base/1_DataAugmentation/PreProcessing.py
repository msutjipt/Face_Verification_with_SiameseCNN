import os
import random
from torchvision import transforms, datasets
from torchvision.utils import save_image
from PIL import Image

class DataAugmentation:
    def __init__(self, image_folder_path):
        self.image_folder_path = image_folder_path

    def create_augmented_picture(self):
        augmentations = [
            (transforms.RandomHorizontalFlip(p=1), 'RandomHorizontalFlip'),
            (transforms.RandomVerticalFlip(p=1), 'RandomVerticalFlip'),
            (transforms.RandomRotation(degrees=(-10, 10)), 'RandomRotation'),
            (transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 'ColorJitter'),
            (transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), 'RandomAffine'),
            (transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=1), 'GaussianBlur'),
            (transforms.RandomApply([transforms.RandomErasing(p=1, scale=(0.02, 0.2))], p=0.5), 'RandomErasing')
        ]

        for root, _, files in os.walk(self.image_folder_path):
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    image = Image.open(image_path).convert('RGB')
                    chosen_augmentation, augmentation_name = random.choice(augmentations)
                    #image = transforms.ToTensor()
                    transformation = transforms.Compose([transforms.ToTensor(), chosen_augmentation])
                    augmented_image = transformation(image)
                    base, ext = os.path.splitext(file)
                    new_file_name = f"{base}_{augmentation_name}{ext}"
                    augmented_image_path = os.path.join(root, new_file_name)
                    save_image(augmented_image, augmented_image_path)
                    print(f"Applied augmentation '{augmentation_name}' and saved augmented image {augmented_image_path}")


if __name__ == "__main__":
    folder_path = "Path_To_Your_Images"
    data_augmentor = DataAugmentation(folder_path)
    data_augmentor.create_augmented_picture()
    print("Finished Augmentation")


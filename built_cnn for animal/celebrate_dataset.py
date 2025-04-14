from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torchvision.transforms import ToTensor,Resize,Compose

class Animal(Dataset):
    def __init__(self,root,transform = None,train = True):
        self.root = root
        self.transform = transform
        self.categoris = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
        self.images_path = []
        self.labels = []

        for i,category in enumerate(self.categoris):
            data_file_path = os.path.join(self.root,category)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path,file_name)
                self.images_path.append(file_path)
                self.labels.append(i)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image,label
if __name__ == '__main__':
    transform = Compose([
        Resize((200,200)),
        ToTensor()
    ])
    dataset = Animal(root='raw-img',transform=transform)
    training_dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    for image,label in training_dataloader:
        print(image.shape)
        print(label)

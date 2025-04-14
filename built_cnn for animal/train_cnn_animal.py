import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from argparse import ArgumentParser
from os import mkdir
import torch.nn as nn
import torch.optim
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from torch.utils.data import DataLoader
from torch.utils .tensorboard import SummaryWriter # thư viện con pytorch giúp lưu thông tin quá trình train
import shutil

from built_cnn_animal import SimpleNeuralNetWork,SimpleCNN
from celebrate_dataset import Animal
from tqdm import tqdm  # tạo ra thanh theo dõi trong quá trình train
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor,Resize,Compose,RandomAffine,ColorJitter,Normalize
from sklearn.model_selection import train_test_split


def get_args():

    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str, default='./data', help="Number of epochs")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="batch size")
    # parser.add_argument("--image_size", "-i", type=int, default=224, help="image size")
    parser.add_argument("--logging", "-l", type=str, default='tensorboard')
    parser.add_argument("--trained_model", "-t", type=str, default='trained_model')
    parser.add_argument("--checkpoint", "-c", type=str, default=None)

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


if __name__ == '__main__':
    args = get_args()
    num_epochs = 100
    train_transform = Compose([
        RandomAffine(    #thực hiện 1 số phép biến đổi ảnh để tăng độ đa dạng của dataset
            degrees=(-5,5),   # phép xoay từ -5 độ đến 5 độ
            translate= (0.15,0.15), # phép dịch
            scale=(0.85,1.15), #phóng to or thu nhỏ ảnh
            shear=5 # làm nghiêng hình ảnh theo trục trái or phải

        ),
        #thay đổi thuộc tính về màu của ảnh
        ColorJitter(
            brightness= 0.5,
            saturation=0.1,
            contrast=0.2,
            hue=0.2

        ),
        Resize((200, 200)),
        ToTensor(),
        # Normalize(mean=[0.485,0.456,0.406],
        #           std = [0.229,0.224,0.225],
        #           )  #chuẩn hóa đầu vào khi dùng transfer learning
    ])



    test_transform = Compose([
        Resize((200, 200)),
        ToTensor()
    ])
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train_set = Animal(root='data_split\\train', transform=train_transform)
    test_set = Animal(root='data_split\\test', transform=test_transform)

    # img, _ = train_set.__getitem__(1455)
    # '''
    # thực hiện chuyển kênh từ C,H,W -> H,W,C và [0,1] ->[0,255] (pil vs opencv đều nhân 255) đồng
    # thời chuyển về dạng uint8 để dùng opencv đọc ảnh
    # '''
    # img = (torch.permute(img,(1,2,0))*255).numpy().astype(np.uint8)
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) # chuyển kênh màu RGB -> BGR (poen cv đọc aảnh khác PIL)
    # cv2.imshow("test img",img)
    # cv2.waitKey(0)
    # exit(0)

    train_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=train_set,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=test_set,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_model):
        mkdir(args.trained_model)
    writer = SummaryWriter(args.logging) # dùng args khởi tạo folder tensor board
    num_interation = len(train_loader)
    model = SimpleCNN(num_classes=10).to(device) # thêm to device kiểm tra có dùng đc trên gpu hay k


    criterion = nn.CrossEntropyLoss()  # định nghĩa trong pytorch hay dùng criterion làm biến thể hiện cho loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # dùng để kiểm tra xem có lưu checkpoit trước đó không, nếu có thì tiếp tục train từ epoch tiếp theo
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        best_accuracy = 0
        start_epoch = 0

    # đẩy model sang train trên gpu
    # if torch.cuda.is_available():
    #     model.cuda()

    for epoch in range(start_epoch,args.epochs):
        model.train()
        progress_bar = tqdm(train_loader,colour="green")
        for inter, (images, labels) in enumerate(progress_bar):
            #
            # if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)

            # forward process
            outputs = model(images)
            loss_values = criterion(outputs, labels)
            progress_bar.set_description("Epoch {}/{}. Interation {}/{}. Loss {:.3f}".format(epoch+1,num_epochs,inter+1,num_interation,loss_values))
            writer.add_scalar('Train/Loss',loss_values,epoch*num_epochs + inter)
            # backward
            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()
        model.eval()
        all_predictions = []
        all_labels = []
        for inter, (images, labels) in enumerate(test_loader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)  # kích thước của predictions có dạng 64x10 (64 hàng, 10 cột)
                indices = torch.argmax(predictions.cpu(), dim=1)  # hàm argmax return về kết quả gồm vị trí
                # của gíá trị max trong prediction và giá trị của nó dim tương đương vs axis 0:cột 1:hàng
                all_predictions.extend(indices)
                loss_values = criterion(predictions, labels)

        all_predictions = [predictions.item() for predictions in all_predictions]
        all_labels = [labels.item() for labels in all_labels]

        # print(all_labels)
        # print('--------------------------------------------------------------')
        # print(all_predictions)
        # exit(0)
        category =['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
        plot_confusion_matrix(writer,confusion_matrix(all_labels,all_predictions),class_names=category,epoch=epoch)
        accuracy = accuracy_score(all_labels,all_predictions)

        print('Epoch {}, Accuracy:{}'.format(epoch + 1,accuracy))
        writer.add_scalar('Val/Accuracy',accuracy,epoch)
        # print(classification_report(all_labels, all_predictions))
        # torch.save(model.state_dict(),'{}/last_cnn.pt'.format(args.trained_model))
        check_point = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(check_point,'{}/last_cnn.pt'.format(args.trained_model))

        if accuracy > best_accuracy:
            check_point = {
                "epoch": epoch + 1,
                "best_acc": best_accuracy,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(check_point, '{}/best_cnn.pt'.format(args.trained_model))
            best_accuracy = accuracy


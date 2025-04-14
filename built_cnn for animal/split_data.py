import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets

# Đường dẫn thư mục dữ liệu gốc
dataset_path = "raw-img"  # 🔹 Thay bằng đường dẫn thư mục chứa ảnh
output_base = "data_split"    # 🔹 Thư mục chứa train/test

# Load dữ liệu từ folder
dataset = datasets.ImageFolder(root=dataset_path)

# Lấy danh sách tệp và nhãn
filepaths = [sample[0] for sample in dataset.imgs]  # Danh sách đường dẫn ảnh
labels = [sample[1] for sample in dataset.imgs]  # Danh sách nhãn class

# Chia dữ liệu thành train (80%) và test (20%)
train_files, test_files, train_labels, test_labels = train_test_split(
    filepaths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Hàm di chuyển ảnh vào thư mục train/test
def move_files(files, labels, target_dir):
    for file, label in zip(files, labels):
        class_name = dataset.classes[label]  # Lấy tên class từ index
        target_path = os.path.join(target_dir, class_name)
        os.makedirs(target_path, exist_ok=True)
        shutil.move(file, os.path.join(target_path, os.path.basename(file)))

# Tạo thư mục train/test và di chuyển ảnh
move_files(train_files, train_labels, os.path.join(output_base, "train"))
move_files(test_files, test_labels, os.path.join(output_base, "test"))

print("✅ Chia dữ liệu thành công!")

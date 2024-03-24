import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize
from tqdm import tqdm
import time

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

# 数据集定义
class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, filenames, transform=None, target_transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.filenames = filenames
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.filenames[idx])
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask



# 模型组件定义
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class conv3x3_block_x1(nn.Module):
    '''(conv => BN => ReLU) * 1'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class conv3x3_block_x2(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)


class conv3x3_block_x3(nn.Module):
    '''(conv => BN => ReLU) * 3'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)


class upsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(upsample, self).__init__()
        self.conv1x1 = conv1x1(in_ch, out_ch)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1x1(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


# FCN模型定义
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.block1 = conv3x3_block_x2(3, 64)
        self.block2 = conv3x3_block_x2(64, 128)
        self.block3 = conv3x3_block_x3(128, 256)
        self.block4 = conv3x3_block_x3(256, 512)
        self.block5 = conv3x3_block_x3(512, 512)
        self.upsample1 = upsample(512, 512, 2)
        self.upsample2 = upsample(512, 256, 2)
        self.upsample3 = upsample(256, num_classes, 16)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)  # Ensure output size matches input
        return x


# 数据预处理和数据集准备
transform = Compose([
    Resize((512, 512)),  # 调整掩膜尺寸
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 512)),  # 调整掩膜尺寸以匹配模型输出
    ToTensor(),
])

image_filenames = [f"{i}.png" for i in range(1, 2329)]

# 假设已经定义了image_filenames
train_filenames, test_filenames = train_test_split(image_filenames, test_size=0.2, random_state=42)

train_dataset = CustomDataset(
    image_folder='Buildingsample_pic',
    mask_folder='Buildingsample_binarymask',
    filenames=train_filenames,
    transform=transform,
    target_transform=target_transform  # 这里传递掩膜的变换
)

test_dataset = CustomDataset(
    image_folder='Buildingsample_pic',
    mask_folder='Buildingsample_binarymask',
    filenames=test_filenames,
    transform=transform,
    target_transform=target_transform  # 这里传递掩膜的变换
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# 模型训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FCN(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100  # Total number of epochs
update_interval = 3  # 更新间隔，单位为秒，这里设置为5分钟
for epoch in range(1, num_epochs + 1):
    model.train()
    start_time = time.time()  # 开始时间
    last_update_time = start_time  # 上次更新时间
    train_loop = tqdm(train_loader, leave=True)
    for images, masks in train_loop:
        current_time = time.time()  # 当前时间
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        if (current_time - last_update_time) > update_interval:
            # 更新进度条
            train_loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            train_loop.set_postfix(loss=loss.item())
            last_update_time = current_time  # 重置上次更新时间

    if epoch % 25 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
        print(f'Epoch {epoch}, Model Saved.')

    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 训练完成后保存最终模型状态
torch.save(model.state_dict(), 'final_model.pth')
print('Training completed and final model saved.')


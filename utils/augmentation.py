import albumentations as album
#定义数据增强
#在训练集上的数据增强包括水平、垂直、镜像
def get_training_augmentation(crop_size):
    train_transform = [
        album.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],p=0.7   ),]
    return album.Compose(train_transform)
#测试集上的数据增强包括对数据进行填充，
def get_validation_augmentation(val_size):
    # 确保网络能被32整除
    test_transform = [
        album.PadIfNeeded(min_height=val_size, min_width=val_size, always_apply=True, border_mode=0),
        album.RandomCrop(height=val_size, width=val_size, always_apply=True),
        ]
    return album.Compose(test_transform)
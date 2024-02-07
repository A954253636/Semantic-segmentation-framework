from utils.augmentation import *
from utils.Dataset import *
from utils.LabelProcessor import *
from utils.Metrics import *
import numpy as np
import matplotlib.pyplot as plt
import config
if __name__ == '__main__':
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    my_train = MyDataset([config.TRAIN_ROOT, config.TRAIN_LABEL], get_training_augmentation(config.crop_size))
    my_val = MyDataset([config.VAL_ROOT, config.VAL_LABEL], get_validation_augmentation(config.val_size))

    train_data = DataLoader(my_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_data = DataLoader(my_val, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    # 确定使用模型
    model = config.usemodel  # ✨确定模型
    # 如果有多个GPU可用，使用DataParallel包装模型
    if t.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
    model = model.cuda()
    # 损失计算方式
    criterion = nn.NLLLoss().to(device)  # ✨损失函数#criterion=losses.DiceLoss(mode="multiclass")
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    best = [0]
    loss_list = []
    train_loss_list = []
    eval_loss_list = []
    # 训练轮次
    for epoch in range(1, config.EPOCH_NUMBER + 1):
        print('Epoch is [{}/{}]'.format(epoch, config.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.2
        """
        学习率衰减策略
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        """
        net = model.train()
        train_loss = 0
        train_oa = 0
        train_miou = 0
        train_class_acc = 0
        eval_loss = 0
        eval_oa = 0
        eval_miou = 0
        eval_class_acc = 0
        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample["img"].to(device))
            img_label = Variable(sample["label"].to(device))

            # 训练
            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_oa += eval_metrix["OA"]
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']

            print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))
        print('|epoch_train_loss {: .8f}|'.format(train_loss / len(train_data)))
        train_loss_list.append(train_loss / len(train_data))  # 损失放在列表中
        metric_description = '|Train OA|: {:.5f}|Train Mean IU|: {:.5f}'.format(
            train_oa / len(train_data),
            train_miou / len(train_data),
        )

        print(metric_description)
        # val
        net = model.eval()
        for j, sample in enumerate(val_data):
            valImg = Variable(sample['img'].to(device))
            valLabel = Variable(sample['label'].long().to(device))

            out = net(valImg)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, valLabel.long())
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = valLabel.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = eval_semantic_segmentation(pre_label, true_label)
            eval_oa = eval_metrics['OA'] + eval_oa
            eval_miou = eval_metrics['miou'] + eval_miou

        val_str = ('|Valid Loss|: {:.5f} \n|Valid OA|: {:.5f} \n|Valid Mean IU|: {:.5f} '.format(
            eval_loss / len(val_data),  # 一个epoch上的平均损失
            eval_oa / len(val_data),
            eval_miou / len(val_data)))
        eval_loss_list.append(eval_loss / len(val_data))
        if max(best) <= eval_miou / len(val_data):
            best.append(eval_miou / len(val_data))
            t.save(net.state_dict(), config.best_pth)
        t.save(net.state_dict(), config.last_pth)
        print(val_str)
    # 打印每个epoch的损失
    plt.plot(range(len(train_loss_list)), train_loss_list, label='Train Loss')
    plt.plot(range(len(train_loss_list)), eval_loss_list, label='Val   Loss')
    plt.title(config.title)
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(config.loss_figure)
    print("损失图像已保存!!!")
    # 将数据转换为pandas DataFrame
    loss_data = pd.DataFrame({
        'Train Loss': train_loss_list,
        'Validation Loss': eval_loss_list
    })
    # 保存为CSV文件
    loss_data.to_csv(config.loss_csv, index=False)
    print("损失已保存！！！")
    print("训练完成！！！")
from utils.augmentation import *
from utils.Dataset import *
from utils.LabelProcessor import *
from utils.Metrics import *
import numpy as np
import matplotlib.pyplot as plt
import config
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = config.BATCH_SIZE
my_test = MyDataset([config.TEST_ROOT, config.TEST_LABEL], get_validation_augmentation(config.val_size))
test_data = DataLoader(my_test, batch_size=1, shuffle=False, num_workers=0)

net = config.usemodel
net.eval()
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
else:
    net=net.to(device)
#参数选择
if config.Parameter_selection=="best":
    net.load_state_dict(torch.load(config.best_pth))  # 加载训练好的模型参数
elif config.Parameter_selection=="last":
    net.load_state_dict(torch.load(config.last_pth))  # 加载训练好的模型参数
else:
    print("Error!(请输入best或者last在config.Parameter_selection中)")
    raise("请返回重新输入！")
net = net.to(device)
error = 0
test_mpa = 0
test_miou = 0
test_class_acc = 0
test_oa = 0
test_recall=0
test_f1=0
test_precision=0
test_kappa=0
for i, sample in enumerate(test_data):
    data = Variable(sample['img']).to(device)
    label = Variable(sample['label']).to(device)
    out = net(data)
    out = F.log_softmax(out, dim=1)

    pre_label = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label]

    true_label = label.data.cpu().numpy()
    true_label = [i for i in true_label]

    eval_metrix = eval_semantic_segmentation(pre_label, true_label)
    test_mpa = eval_metrix['mean_class_accuracy'] + test_mpa
    test_miou = eval_metrix['miou'] + test_miou
    test_oa = eval_metrix['OA'] + test_oa
    test_recall=eval_metrix["recall"]+test_recall
    test_f1=eval_metrix["f1"]+test_f1
    test_precision=eval_metrix["precision"]+test_precision
    test_kappa=eval_metrix["kappa"]+test_kappa
    if len(eval_metrix['class_accuracy']) < config.class_num:                  #注意几分类
        eval_metrix['class_accuracy'] = 0
        test_class_acc = test_class_acc + eval_metrix['class_accuracy']
        error += 1
    else:
        test_class_acc = test_class_acc + eval_metrix['class_accuracy']
    print(eval_metrix['class_accuracy'], '================', i)
epoch_str = ('test_miou: {:.5f}, test_accuracy(oa): {:.5f},  test_recall: {:.5f},test_f1:{:.5f},test_precision:{:.5f},test_kappa:{:.5f}'.format(
    test_miou / (len(test_data) - error),
    test_oa / (len(test_data) - error),
    #train_class_acc/(len(test_data)-error),#类别精度
    test_recall / (len(test_data) - error),
    test_f1/ (len(test_data) - error),
    test_precision/ (len(test_data) - error),
    test_kappa/ (len(test_data) - error),
))
with open(config.test_result, 'w') as file:
    file.write(epoch_str)
print(epoch_str+'==========last')
print("测试完成！！！")

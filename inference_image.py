from utils.augmentation import *
from utils.Dataset import *
from utils.LabelProcessor import *
from utils.Metrics import *
import numpy as np
import matplotlib.pyplot as plt
import config
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

my_test = MyDataset([config.TEST_ROOT, config.TEST_LABEL],get_validation_augmentation(config.val_size))
test_data = DataLoader(my_test, batch_size=1, shuffle=False, num_workers =0)

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

pd_label_color = pd.read_csv(config.class_dict_path, sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
    tmp = pd_label_color.iloc[i]
    color = [tmp['r'], tmp['g'], tmp['b']]
    colormap.append(color)
cm = np.array(colormap).astype('uint8')
dir = config.dir              #输出结果
total_inference_time=0
num_images = len(test_data)
for i, sample in enumerate(test_data):
    valImg = sample['img'].to(device)
    valLabel = sample['label'].long().to(device)
    # Measure time taken for inference
    start_time = time.time()
    out = net(valImg)
    out = F.log_softmax(out, dim=1)
    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
    pre = cm[pre_label]
    pre1 = Image.fromarray(pre)
    pre1.save(dir + str(i+1) + '.png')
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time
    print(f'Processed image {i}, Inference time: {inference_time:.5f} seconds')
fps = num_images / total_inference_time
print(f"fps:{fps}")
print('完成喽！！')
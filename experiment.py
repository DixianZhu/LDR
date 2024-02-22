from loss import CELoss, CSLoss, WWLoss, LDRLoss_V1, ALDRLoss_V1, TGCELoss, GCELoss, RLLLoss, MAELoss, MSELoss, SCELoss, JSCLoss, NCEandRCE, NCEandAGCE, NCEandAUL
from optimizers import SGD
from models import ResNet18, FFNN
from datasets import KMNIST, TIMG
from cifar import CIFAR10, CIFAR100
from utils import set_all_seeds, uniform_noise, classDep_noise, classDep_news20, classDep_letter, classDep_vowel
from utils import accuracy, balanced_accuracy, ImageDataset, TabularDataset, ImageDataset_cifar
import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import argparse
parser = argparse.ArgumentParser(description = 'LDR/ALDR experiments')
parser.add_argument('--loss', default='LDR', type=str, help='loss functions to use (LDR, ALDR, CE, CS, WW, etc.)')
parser.add_argument('--dataset', default='letter', type=str, help='the name for the dataset to use (letter, vowel, news20, etc.)')
parser.add_argument('--noise_type', default='class-dependent', type=str, help='noise type imposed to data (uniform, class-dependent)')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum parameter for SGD optimizer')
parser.add_argument('--decay', default=5e-3, type=float, help='weight decay for training the model')
parser.add_argument('--noise_level', default=0.5, type=float, help='the level of noise imposed to the data')
parser.add_argument('--alpha', default=2, type=float, help='alpha parameter for ALDR loss')
parser.add_argument('--margin', default=0.1, type=float, help='margin')

# paramaters
args = parser.parse_args()
SEED = 123
BATCH_SIZE = 64
lr = args.lr
weight_decay = args.decay
set_all_seeds(SEED)
# dataloader
if args.dataset == 'kmnist':
  (train_data, train_label), (test_data, test_label) = KMNIST()
  train_label = train_label.squeeze()
  test_label = test_label.squeeze()
  num_class = 49
  inchannels = 1
  if args.noise_type == 'class-dependent':
    train_label = classDep_noise(train_label, num_class, args.noise_level)
  elif args.noise_type == 'uniform':
    train_label = uniform_noise(train_label, num_class, args.noise_level)
elif args.dataset == 'timg':
  (train_data, train_label), (test_data, test_label) = TIMG()
  train_label = train_label.squeeze()
  test_label = test_label.squeeze()
  num_class = 200
  inchannels = 3
  if args.noise_type == 'class-dependent':
    train_label = classDep_noise(train_label, num_class, args.noise_level)
  elif args.noise_type == 'uniform':
    train_label = uniform_noise(train_label, num_class, args.noise_level)
elif args.dataset == 'letter':
  tmp = np.load('./data/letter_train.npz')
  train_data = tmp['arr_0'].astype(float)
  train_label = np.argmax(tmp['arr_1'], axis=1)
  num_class = 26
  if args.noise_type == 'class-dependent':
    train_label = classDep_letter(train_label, args.noise_level)
  elif args.noise_type == 'uniform':
    train_label = uniform_noise(train_label, num_class, args.noise_level)
  tmp = np.load('./data/letter_test.npz')
  test_data = tmp['arr_0'].astype(float)
  test_label = np.argmax(tmp['arr_1'], axis=1)
elif args.dataset == 'news20':
  tmp = np.load('./data/news20_train.npz')
  train_data = tmp['arr_0'].astype(float)
  train_label = tmp['arr_1']
  num_class = 20
  if args.noise_type == 'class-dependent':
    train_label = classDep_news20(train_label, args.noise_level)
  elif args.noise_type == 'uniform':
    train_label = uniform_noise(train_label, num_class, args.noise_level)
  tmp = np.load('./data/news20_test.npz')
  test_data = tmp['arr_0'].astype(float)
  test_label = tmp['arr_1']
elif args.dataset == 'vowel':
  tmp = np.load('./data/vowel_train.npz', allow_pickle=True)
  train_data = tmp['arr_0'].astype(float)
  train_label = np.argmax(tmp['arr_1'], axis=1)
  num_class = 11
  if args.noise_type == 'class-dependent':
    train_label = classDep_vowel(train_label, args.noise_level)
  elif args.noise_type == 'uniform':
    train_label = uniform_noise(train_label, num_class, args.noise_level)
  tmp = np.load('./data/vowel_test.npz', allow_pickle=True)
  test_data = tmp['arr_0'].astype(float)
  test_label = np.argmax(tmp['arr_1'], axis=1)
elif args.dataset == 'aloi':
  tmp = np.load('./data/aloi_train.npz', allow_pickle=True)
  train_data = tmp['arr_0'].astype(float)
  train_label = tmp['arr_1'].astype(int)
  num_class = 1000
  if args.noise_type == 'class-dependent':
    train_label = classDep_noise(train_label, num_class, args.noise_level)
  elif args.noise_type == 'uniform':
    train_label = uniform_noise(train_label, num_class, args.noise_level)
  tmp = np.load('./data/aloi_test.npz', allow_pickle=True)
  test_data = tmp['arr_0'].astype(float)
  test_label = tmp['arr_1'].astype(int)
elif args.dataset == 'cifar100':
  (train_data, train_label), (test_data, test_label) = CIFAR100()
  num_class = 100
  train_label = train_label.squeeze()
  test_label = test_label.squeeze()
  if args.noise_type == 'class-dependent':
    train_label = classDep_noise(train_label, num_class, args.noise_level)
  elif args.noise_type == 'uniform':
    train_label = uniform_noise(train_label, num_class, args.noise_level)
  inchannels = 3

test_label = uniform_noise(test_label, num_class, 0.0)

if args.dataset == 'timg':
  traindSet = ImageDataset(train_data, train_label, image_size=32, crop_size=60)
  testSet = ImageDataset(test_data, test_label, image_size=32, crop_size=64, mode = 'test')
elif args.dataset in ['letter','news20', 'vowel', 'aloi','kmnist']:
  traindSet = TabularDataset(train_data, train_label)
  testSet = TabularDataset(test_data, test_label)
elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
  traindSet = ImageDataset_cifar(train_data, train_label)
  testSet = ImageDataset_cifar(test_data, test_label, mode = 'test')


kf = KFold(n_splits=5)
tmpX = np.zeros((len(traindSet),1))
best_para = 0 
parameter_set = [1.0]
if args.loss in ['CE', 'CEk', 'MAE', 'MSE']:
  parameter_set = [0.0]
if args.loss in ['CS', 'WW', 'SVMk', 'RLL']:
  parameter_set = [0.1, 1.0, 10.0]
if args.loss in [ 'ALDR', 'LDR']:
  parameter_set = [0.1, 1.0, 10]
if args.loss == 'GCE':
  parameter_set = [0.05, 0.7, 0.95]
if args.loss == 'tGCE':
  parameter_set = [0.05, 0.7, 0.95]
if args.loss == 'SCE':
  parameter_set = [0.05, 0.5, 0.95]
if args.loss == 'JSC':
  parameter_set = ['0.1 0.9','0.5 0.5','0.9 0.1']
if args.loss in ['NCEandRCE', 'NCEandAGCE', 'NCEandAUL']:
  parameter_set = [0.1, 5, 9.9]
testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=0, shuffle=False)
part = 0

print ('Start Training')
print ('-'*30)

for train_id, val_id in kf.split(tmpX):
  tmp_trainSet = torch.utils.data.Subset(traindSet, train_id)
  trainN = len(tmpX)
  tmp_valSet = torch.utils.data.Subset(traindSet, val_id)
  for para in parameter_set:
    trainloader =  torch.utils.data.DataLoader(dataset=tmp_trainSet, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
    validloader =  torch.utils.data.DataLoader(dataset=tmp_valSet, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    if args.dataset in ['kmnist', 'timg', 'cifar100']:
      model = ResNet18(num_classes=num_class, inchannels = inchannels)
      model = model.cuda()
      total_epochs = 30
      decay_epochs = [10, 20]
    else:
      model = FFNN(num_classes=num_class, activation='tanh', dims=train_data.shape[1])
      model = model.cuda()
      total_epochs = 100
      decay_epochs = [50, 75]
      
    # define loss & optimizer
    if args.loss == 'CE':
      Loss = CELoss()
    elif args.loss == 'CS':
      Loss = CSLoss(threshold = para)
    elif args.loss == 'WW':
      Loss = WWLoss(threshold = para)
    elif args.loss == 'SVMk':
      Loss = SVMkLoss(threshold = para, k = args.topk)
    elif args.loss == 'CEk':
      Loss = CEkLoss(k = args.topk)
    elif args.loss == 'LDR':
      Loss = LDRLoss_V1(threshold = args.margin, Lambda = para)
    elif args.loss == 'ALDR':
      if args.dataset != 'cifar100':
        Loss = ALDRLoss_V1(threshold = args.margin, N=trainN, Lambda = para, alpha=args.alpha)
      else:
        Loss = ALDRLoss_V1(threshold = args.margin, N=trainN, Lambda = para, alpha=args.alpha, softplus=True)
    elif args.loss == 'GCE':
      Loss = GCELoss(q = para)
    elif args.loss == 'tGCE':
      Loss = TGCELoss(q = para, k = 0.5, N=trainN)
    elif args.loss == 'RLL':
      Loss = RLLLoss(threshold = para)
    elif args.loss == 'JSC':
      Loss = JSCLoss(num_classes=num_class, weights = para)
    elif args.loss == 'NCEandRCE':
      Loss = NCEandRCE(num_classes=num_class, alpha = para, beta=10-para)
    elif args.loss == 'NCEandAGCE':
      Loss = NCEandAGCE(num_classes=num_class, alpha = para, beta=10-para)
    elif args.loss == 'NCEandAUL':
      Loss = NCEandAUL(num_classes=num_class, alpha = para, beta=10-para)
    elif args.loss == 'MAE':
      Loss = MAELoss()
    elif args.loss == 'MSE':
      Loss = MSELoss()
    elif args.loss == 'SCE':
      Loss = SCELoss(balance = para)
    
    optimizer = SGD(model=model, lr=lr, weight_decay=weight_decay, momentum=args.momentum) 
    print('para=%s, part=%s'%(para, part))
    for epoch in range(total_epochs): # could customize the running epochs
      tr_loss = 0
      if epoch in decay_epochs:
          optimizer.update_stepsize(decay_factor=10)
      for idx, data in enumerate(trainloader):
          train_data, train_label, _, ids = data
          train_data, train_label = train_data.cuda(), train_label.cuda()
          y_pred = model(train_data.float()) 
          if args.loss == 'CE':
            loss = Loss(y_pred, train_label.to(dtype=torch.long))
          elif args.loss in ['ALDR','tGCE']:
            if args.loss == 'tGCE' and (epoch+1)%10 ==0 and epoch > int(total_epochs/2):
              Loss.prune(y_pred, torch.nn.functional.one_hot(torch.squeeze(train_label.to(dtype=torch.long)), num_classes=num_class), ids)
              continue
            else:
              loss = Loss(y_pred, torch.nn.functional.one_hot(torch.squeeze(train_label.to(dtype=torch.long)), num_classes=num_class), ids) 
          else:
            loss = Loss(y_pred, torch.nn.functional.one_hot(torch.squeeze(train_label.to(dtype=torch.long)), num_classes=num_class))
          tr_loss = tr_loss  + loss.cpu().detach().numpy()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      tr_loss = tr_loss/idx
      print ('Epoch=%s, BatchID=%s, training_loss=%.4f, lr=%.4f'%(epoch, idx, tr_loss,  optimizer.lr)) 
      model.eval()
      tr_loss = 0
      with torch.no_grad(): 
        test_pred = []
        test_true = [] 
        val_pred = []
        val_true = [] 
        for jdx, data in enumerate(testloader):
          if True:
            test_data, test_label, _, _ = data
            test_data = test_data.cuda()
            y_pred = model(test_data.float())
            y_pred = torch.nn.functional.softmax(y_pred,dim=1)
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(test_label.numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        if args.dataset in ['kmnist']:
          test_acc =  balanced_accuracy(torch.from_numpy(test_pred), torch.from_numpy(test_true), num_class, topk=[1,2,3,4,5]) 
        else:
          test_acc =  accuracy(torch.from_numpy(test_pred), torch.from_numpy(test_true), topk=[1,2,3,4,5]) 

        for jdx, data in enumerate(validloader):
          if True:
            val_data, val_label, _, _ = data
            val_data = val_data.cuda()
            y_pred = model(val_data.float())
            y_pred = torch.nn.functional.softmax(y_pred,dim=1)
            val_pred.append(y_pred.cpu().detach().numpy())
            val_true.append(val_label.numpy())
        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        if args.dataset in ['kmnist']:
          val_acc =  balanced_accuracy(torch.from_numpy(val_pred), torch.from_numpy(val_true), num_class, topk=[1,2,3,4,5]) 
        else:
          val_acc =  accuracy(torch.from_numpy(val_pred), torch.from_numpy(val_true), topk=[1,2,3,4,5]) 
        model.train()
        print ('valtop1=%.4f, valtop2=%.4f, valtop3=%.4f, valtop4=%.4f, valtop5=%.4f'%(val_acc[0], val_acc[1], val_acc[2], val_acc[3], val_acc[4]))
        print ('testtop1=%.4f, testtop2=%.4f, testtop3=%.4f, testtop4=%.4f, testtop5=%.4f'%(test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4]))
  part += 1 


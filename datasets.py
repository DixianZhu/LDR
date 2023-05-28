from tensorflow.python.keras.utils.data_utils import get_file
import os 
import numpy as np

IMG_SIZE = 128

def load_data(path):
    tr_data_path = os.path.join(path,'TIMG_train_X.npy')
    tr_label_path = os.path.join(path, 'TIMG_train_Y.npy')
    te_data_path = os.path.join(path,'TIMG_test_X.npy')
    te_label_path = os.path.join(path, 'TIMG_test_Y.npy')
    train_data = np.load(tr_data_path)
    train_label = np.expand_dims(np.load(tr_label_path),axis=1).astype(int)
    N = train_label.shape[0]
    np.random.seed(0)
    randids = np.random.permutation(N)
    train_data = train_data[randids]
    train_label = train_label[randids]
    test_data = np.load(te_data_path)
    test_label = np.expand_dims(np.load(te_label_path),axis=1).astype(int)
    print(train_data.shape)
    print(test_data.shape)
    return train_data, train_label, test_data, test_label

def TIMG():
    
    
    train_X, train_Y,  test_X, test_Y = load_data('../data/')

    # convert data type
    train_X, train_Y = train_X.astype(float), train_Y.astype(np.int32)
    test_X, test_Y = test_X.astype(float), test_Y.astype(np.int32)
    
    return  (train_X, train_Y), (test_X, test_Y)


def KMNIST():
  tmp = np.load('../data/k49-train-imgs.npz')
  trX = np.expand_dims(tmp['arr_0'],axis=1)
  tmp = np.load('../data/k49-train-labels.npz')
  trY = tmp['arr_0']
  tmp = np.load('../data/k49-test-imgs.npz')
  teX = np.expand_dims(tmp['arr_0'],axis=1)
  tmp = np.load('../data/k49-test-labels.npz')
  teY = tmp['arr_0']
  print(trX.shape)
  print(teX.shape)
  return (trX, trY), (teX, teY) 


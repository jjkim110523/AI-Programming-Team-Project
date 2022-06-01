import torch


from mnist_classification.data_loader import load_train, load_test

from mnist_classification.models.fc_model import FullyConnectedClassifier
from mnist_classification.models.cnn_model import ConvolutionalClassifier
from train import get_model
from mnist_classification.data_loader import get_loaders


model_fn = "./model.pth"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load(fn, device):
    d = torch.load(fn, map_location=device)
    
    return d['config'], d['model']

def test(model, x, y, to_be_shown=True):
    model.eval()
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    classes = list(0. for i in range(10))

    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))
        
        accuracy = correct_cnt / total_cnt
        print("Accuracy: %.4f" % accuracy)



train_config, state_dict = load(model_fn, device)

model = get_model(train_config).to(device)
model.load_state_dict(state_dict)

# Accuracy
x, y = load_test(flatten=True if train_config.model == 'fc' else False)

# if model == 'fc'
x, y = x.to(device), y.to(device)

# if model == 'cnn'
# x, y = x.to(device).reshape(-1, 28, 28), y.to(device)

test(model, x, y, to_be_shown=True)

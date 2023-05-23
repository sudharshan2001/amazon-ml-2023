import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from dnn_model import DNNModel
from utils import mape_obj1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = np.load("./processed files/train.npz")
test = np.load("./processed files/test.npz")

X_train = train['a']
y_train = train['b']
X_test = test['a']
y_test = test['b']

batch_size = 32

class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):

        x = torch.from_numpy(self.data[index]).float()

        y = torch.tensor(self.target[index]).float()
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.data)


train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

del X_train, y_train, X_test, y_test

model = DNNModel(301, [1024,512,256],1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = torch.nn.MSELoss()
step_size = 30
gamma = 0.1
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

epochs = 10
for epoch in range(epochs):
    train_loss = 0.0
    test_eval_  = 0.0
    
    model.train()
    for inputs, labels in tqdm(train_loader):
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        eval_ = mape_obj1(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
        test_eval_ += eval_
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
        
    train_loss /= len(train_loader)
    test_eval_ /= len(train_loader)
    
    scheduler.step()
    
    test_loss = 0.0
    test_crit = 0.0
    
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            output = model(data)
            crit_ = criterion(output.squeeze(), target.float())
            loss = mape_obj1(output.detach().cpu().numpy(), target.detach().cpu().numpy())
            test_loss += loss
            test_crit += crit_.item()
            
    test_crit /= len(test_loader)
    test_loss  /= len(test_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f},Train Epoch: {test_eval_:.4f} Test Loss {test_crit:.4f} Test Eval: {test_loss:.4f}')


model.eval()
with torch.no_grad():
    running_loss = 0.0
    for inputs, labels in tqdm(test_loader):
        outputs = model(**inputs)
        loss = criterion(outputs.logits.squeeze(), labels.float())
        running_loss += loss.item() * inputs['input_ids'].size(0)
    test_loss = running_loss / len(test_dataset)
    print('Test Loss: {:.4f}'.format(test_loss))
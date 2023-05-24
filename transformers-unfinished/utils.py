import torch
from tqdm import tqdm, trange
from sklearn import metrics

def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in range(epochs):
        print(epoch)
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()  
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(output, target.type_as(output))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Training loss  {train_loss/len(train_loader)}")
        val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
        print("Epoch {} Validation Loss : {}".format(epoch, val_loss))
    return model

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
            
            mean_loss += criterion(output, target.type_as(output)).item()
#             mean_err += get_rmse(output, target)
            count += 1
            
    return mean_loss/count

def get_rmse(output, target):
    err = torch.sqrt(metrics.mean_squared_error(target, output))
    return err

def predict(model, dataloader, device):
    predicted_label = []
    actual_label = []
    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
                        
            predicted_label += output
            actual_label += target
            
    return predicted_label

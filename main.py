import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

# import things I modify
from model.go_res import GO_Res, save_model_stru
from data_loader import load_data

torch.cuda.set_device(1) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch = 100
gamma = 0.7

model = GO_Res().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
# save_model_stru('model/resnet.oonx', model, (1, 13, 9, 9)) # save to oonx for reviewing model 


epochs = 100
batch_size = 1000
steps, training_loss, testing_loss = 0, 0, 0
print_every = 20
train_losses, test_losses = [], []

train_loader, test_loader = load_data(batch_size)

for epochs in range(1, epochs + 1):
    scheduler.step()
    for i, data in enumerate(train_loader):
        # model.train()
        steps += 1
        xs, ys = data
        ys = torch.reshape(ys, (-1,))
        xs = torch.tensor(xs, dtype=torch.float, device=device) # long type for CrossEntropyLoss
        ys = torch.tensor(ys, dtype=torch.long, device=device)
        optimizer.zero_grad()
        logps = model.forward(xs)
        _, pred = torch.max(logps, dim=1)
        # print("pred:", pred[0:10], "\nground_Truth:", ys[0:10]) # check pred status
        loss = criterion(logps, ys)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        print("step:", steps, "lr:", scheduler.get_lr())
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    xs_, ys_ = data
                    ys_ = torch.reshape(ys_, (-1,))
                    xs_ = torch.tensor(xs_, dtype=torch.float, device=device)
                    ys_ = torch.tensor(ys_, dtype=torch.long, device=device)
                    logps = model.forward(xs_)
                    batch_loss = criterion(logps, ys_)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == ys_.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(training_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))                    
            print(f"Epoch {epoch+1}/{epochs}.. " f"Train loss: {training_loss/print_every:.3f}.. "f"Test loss: {test_loss/len(test_loader):.3f}.. "f"Test accuracy: {accuracy/len(test_loader):.3f}")
            training_loss = 0
            
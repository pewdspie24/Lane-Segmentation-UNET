batch_size = 16
LR = 1e-4
epochs = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainDataset = CustomDataset('./final/train/')
trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=2)

validDataset = CustomDataset('./final/validate/')
validLoader = DataLoader(validDataset, batch_size=batch_size, num_workers=2)

model = UNet(3,1).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0)
criterion = nn.BCEWithLogitsLoss()

#train
losses_train = []
losses_valid = []
for epoch in range(epochs):
    running_train_loss = 0.0
    running_valid_loss = 0.0
    
    model.train()
    for (input, mask) in tq.tqdm(trainLoader): #for colab
    #for (input, mask) in tqdm(trainLoader): #for pc
        optimizer.zero_grad()
        input = input.to(device)
        mask = mask.to(device)
        output = model(input)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * input.size(0) 
    epoch_train_loss = running_train_loss / len(trainLoader)
    losses_train.append(epoch_train_loss)
    print("")
    print('Training, Epoch {} - Loss {}'.format(epoch+1, epoch_train_loss))

    model.eval()
    with torch.no_grad():
      for (input, mask) in tq.tqdm(validLoader): #for colab
      #for (input, mask) in tqdm(validLoader): #for pc
        input = input.to(device)
        mask = mask.to(device)
        output = model(input)
        loss = criterion(output, mask)
        running_valid_loss += loss.item() * input.size(0)
      epoch_valid_loss = running_valid_loss / len(validLoader)
      losses_valid.append(epoch_valid_loss)
      print("")
      print('Validating, Epoch {} - Loss {}'.format(epoch+1, epoch_valid_loss))
plt.plot(losses_train, label="train")
plt.plot(losses_valid, label="valid") 
plt.legend(loc="upper left")
torch.save(model.state_dict(), "/content/drive/MyDrive/AI-ML/Lane_Segment_PyTorch/checkpointLaneSegmentTemp.pth")
def tuning(net, data_loader, optimizer, loss_func, device):

    n = len(data_loader.dataset)
    net.train()
    train_loss = []

    for batch_idx, (image, mask) in enumerate(data_loader):
        optimizer.zero_grad()

        image = image.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)

        masks, ensemble_mask = net(image)

        loss = loss_func(masks, ensemble_mask, mask)
        loss.backward()

        optimizer.step()
        batch_size = image.shape[0]
        train_loss.append(loss.item() * batch_size)

    total_loss = sum(train_loss) / n

    return train_loss, total_loss

def validation(net, data_loader, optimizer, loss_func, device):

    n = len(data_loader.dataset)
    net.eval()
    val_loss = []

    with torch.no_grad():
        for batch_idx, (image, mask) in enumerate(data_loader):

            image = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)
            
            masks, ensemble_mask = net(image)

            loss = loss_func(masks, ensemble_mask, mask)
            batch_size = image.shape[0]
            val_loss.append(loss.item() * batch_size)

    total_loss = sum(val_loss) / n

    return val_loss, total_loss  

def train(net, device, epochs, lr, dataset, batch_size, loss_func=nn.BCEWithLogitsLoss(), save_model_path='path to save your model'):

    net.to(device=device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')

    for epoch in range(epochs):
        print("="*10 + f" Epoch {epoch+1} " + "="*10) 

        train_history, train_loss = tuning(net, data_loader, optimizer, loss_func, device)    
        print(f"train_loss: {train_loss}")    

        val_history, val_loss = validation(net, data_loader, optimizer, loss_func, device)
        print(f"val_loss: {val_loss}") 

        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), save_model_path)

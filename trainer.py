import torch

def train_kc_model(model, optimizer, criterion, train_loader, val_loader, epochs, device): # TODO Scheduler
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

        print("Metrics for the last batch :")
        report_metrics(output.detach().cpu(), labels.detach().cpu(), epoch, loss.detach().cpu())
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
            print("Metrics for the last batch :")
            report_metrics(output.detach().cpu(), labels.detach().cpu(), epoch, loss.detach().cpu())

def report_metrics(output, labels, epoch, loss):
    accuracy = output.argmax(dim=1).eq(labels.argmax(dim=1)).sum() / labels.shape[0]
    print("Epoch : %d, Loss : %.2f, Accuracy : %.2f" 
                  %(epoch+1, loss, accuracy), end="") 
    N_pos_classified = output.argmax(dim=1).eq(1).sum() > 0
    if(N_pos_classified > 0):
        precision = torch.logical_and(output.argmax(dim=1).eq(1), labels.argmax(dim=1).eq(1)).sum() / N_pos_classified
        print(", Precision : %.2f"%precision, end="")
    else:
        print(", Precision : No Data", end="")
    N_pos = labels.argmax(dim=1).eq(1).sum() > 0
    if(N_pos > 0):
        recall = torch.logical_and(output.argmax(dim=1).eq(1), labels.argmax(dim=1).eq(1)).sum() / N_pos
        print(", Recall : %.2f"%recall, end="")
    else:
        print(", Recall : No Data", end="")
    print("")
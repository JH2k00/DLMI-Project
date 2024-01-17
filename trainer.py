import torch

def train_kc_model(model, optimizer, criterion, train_loader, val_loader, epochs, device): # TODO Scheduler
    N_train = len(train_loader.dataset)
    N_val = len(val_loader.dataset)
    for epoch in range(epochs):
        model.train()

        TP = 0
        P = 0
        PP = 0
        accuracy = 0
        epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            TP += torch.logical_and(output.argmax(dim=1).eq(1), labels.argmax(dim=1).eq(1)).sum()
            PP += output.argmax(dim=1).eq(1).sum()
            P += labels.argmax(dim=1).eq(1).sum()
            accuracy += output.argmax(dim=1).eq(labels.argmax(dim=1)).sum()
            epoch_loss += loss

            loss.backward()
            optimizer.step()

        accuracy /= N_train
        precision = TP / PP
        recall = TP / P
        epoch_loss /= len(train_loader)
        print("Training :")
        print("Epoch : %d, Loss : %.2f, Accuracy : %.2f, Precision : %.2f, Recall : %.2f" 
                  %(epoch+1, loss, accuracy, precision, recall)) 
        
        model.eval()
        with torch.no_grad():
            TP = 0
            P = 0
            PP = 0
            accuracy = 0
            epoch_loss = 0

            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)

                TP += torch.logical_and(output.argmax(dim=1).eq(1), labels.argmax(dim=1).eq(1)).sum()
                PP += output.argmax(dim=1).eq(1).sum()
                P += labels.argmax(dim=1).eq(1).sum()
                accuracy += output.argmax(dim=1).eq(labels.argmax(dim=1)).sum()
                epoch_loss += loss
            
            accuracy /= N_val
            precision = TP / PP
            recall = TP / P
            epoch_loss /= len(val_loader)
            print("Validating :")
            print("Epoch : %d, Loss : %.2f, Accuracy : %.2f, Precision : %.2f, Recall : %.2f" 
                    %(epoch+1, loss, accuracy, precision, recall))
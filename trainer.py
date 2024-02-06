import torch
import gc
import os

def train_kc_model_sam(model, optimizer, criterion, train_loader, val_loader, model_name, epochs, device): # TODO Scheduler
    N_train = len(train_loader.dataset)
    N_val = len(val_loader.dataset)
    best_f1 = 0
    for epoch in range(epochs):
        model.train()

        TP = 0
        P = 0
        PP = 0
        accuracy = 0
        epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            torch.cuda.empty_cache()
            gc.collect()
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output, labels)
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # TODO Try different values and Scheduler
            optimizer.first_step(zero_grad=True)

            output = model(images)

            loss = criterion(output, labels)
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # TODO Try different values and Scheduler
            optimizer.second_step(zero_grad=True)

            TP += torch.logical_and(output.argmax(dim=1).eq(1), labels.argmax(dim=1).eq(1)).sum()
            PP += output.argmax(dim=1).eq(1).sum()
            P += labels.argmax(dim=1).eq(1).sum()
            accuracy += output.argmax(dim=1).eq(labels.argmax(dim=1)).sum()
            epoch_loss += loss

        accuracy = accuracy.detach().cpu() / N_train
        precision = TP.detach().cpu() / PP.detach().cpu()
        recall = TP.detach().cpu() / P.detach().cpu()
        epoch_loss = epoch_loss.detach().cpu()
        print("Training :")
        print("Epoch : %d, Loss : %.3f, Accuracy : %.3f, Precision : %.3f, Recall : %.3f" 
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

                TP += torch.logical_and(output.argmax(dim=1).eq(1), labels.argmax(dim=1).eq(1)).sum().detach().cpu()
                PP += output.argmax(dim=1).eq(1).sum().detach().cpu()
                P += labels.argmax(dim=1).eq(1).sum().detach().cpu()
                accuracy += output.argmax(dim=1).eq(labels.argmax(dim=1)).sum().detach().cpu()
                epoch_loss += loss
            
            accuracy = accuracy / N_val
            precision = TP / PP
            recall = TP / P
            epoch_loss = epoch_loss
            print("Validating :")
            print("Epoch : %d, Loss : %.3f, Accuracy : %.3f, Precision : %.3f, Recall : %.3f" 
                    %(epoch+1, loss, accuracy, precision, recall))
        
        f1 = 2*precision*recall / (precision + recall)
        if(f1 > best_f1):
            best_f1 = f1
            checkpoint = {"model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join("saved_models", "%s_best.pt"%model_name))
            print('Model with best f1 score:{} is stored at folder:{}'.format(best_f1, 'saved_models/'+'%s_best.pt'%model_name))
        checkpoint = {"model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()}
        torch.save(checkpoint, os.path.join("saved_models", "%s.pt"%model_name))
        print('Model is stored at folder:{}'.format('saved_models/'+'%s.pt'%model_name))

def train_kc_model(model, optimizer, criterion, train_loader, val_loader, scaler, model_name, epochs, device): # TODO Scheduler
    N_train = len(train_loader.dataset)
    N_val = len(val_loader.dataset)
    best_f1 = 0
    for epoch in range(epochs):
        model.train()

        TP = 0
        P = 0
        PP = 0
        accuracy = 0
        epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            torch.cuda.empty_cache()
            gc.collect()
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type='cuda'):
                output = model(images)
                loss = criterion(output, labels)

            TP += torch.logical_and(output.argmax(dim=1).eq(1), labels.argmax(dim=1).eq(1)).sum()
            PP += output.argmax(dim=1).eq(1).sum()
            P += labels.argmax(dim=1).eq(1).sum()
            accuracy += output.argmax(dim=1).eq(labels.argmax(dim=1)).sum()
            epoch_loss += loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # TODO Try different values and Scheduler
            scaler.step(optimizer)
            scaler.update()

        accuracy = accuracy.detach().cpu() / N_train
        precision = TP.detach().cpu() / PP.detach().cpu()
        recall = TP.detach().cpu() / P.detach().cpu()
        epoch_loss = epoch_loss.detach().cpu()
        print("Training :")
        print("Epoch : %d, Loss : %.3f, Accuracy : %.3f, Precision : %.3f, Recall : %.3f" 
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
                with torch.autocast(device_type='cuda'):
                    output = model(images)
                    loss = criterion(output, labels)

                TP += torch.logical_and(output.argmax(dim=1).eq(1), labels.argmax(dim=1).eq(1)).sum().detach().cpu()
                PP += output.argmax(dim=1).eq(1).sum().detach().cpu()
                P += labels.argmax(dim=1).eq(1).sum().detach().cpu()
                accuracy += output.argmax(dim=1).eq(labels.argmax(dim=1)).sum().detach().cpu()
                epoch_loss += loss
            
            accuracy = accuracy / N_val
            precision = TP / PP
            recall = TP / P
            epoch_loss = epoch_loss
            print("Validating :")
            print("Epoch : %d, Loss : %.3f, Accuracy : %.3f, Precision : %.3f, Recall : %.3f" 
                    %(epoch+1, loss, accuracy, precision, recall))
        f1 = 2*precision*recall / (precision + recall)
        if(f1 > best_f1):
            best_f1 = f1
            checkpoint = {"model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join("saved_models", "%s_best.pt"%model_name))
            print('Model with best f1 score:{} is stored at folder:{}'.format(best_f1, 'saved_models/'+'%s_best.pt'%model_name))
        
        checkpoint = {"model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()}
        torch.save(checkpoint, os.path.join("saved_models", "%s.pt"%model_name))
        print('Model is stored at folder:{}'.format('saved_models/'+'%s.pt'%model_name))
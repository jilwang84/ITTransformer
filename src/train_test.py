# Copyright (c) 2023-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import torch
from tqdm import tqdm


# Train function
def train(model, train_dataloader, test_dataloader, device, optimizer, scheduler, epochs, criterion, early_stopping_patience, logger, model_save_file):

    training_loss = []
    training_epoch = []
    training_acc = []
    testing_acc = []
    best_acc = 0
    early_stopping_count = 0
    train_dataloader_len = len(train_dataloader)
    if early_stopping_patience is None:
        early_stopping_patience = epochs + 1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        tbar = tqdm(train_dataloader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_categ, x_numer, y, x_categ_mask, x_numer_mask) in enumerate(tbar):
            optimizer.zero_grad()
            x_categ = x_categ.to(device)
            x_numer = x_numer.to(device)
            x_categ_mask = x_categ_mask.to(device)
            x_numer_mask = x_numer_mask.to(device)
            y = torch.squeeze(y).to(device)

            y_pred = model(x_categ, x_numer, x_categ_mask, x_numer_mask)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # Set prefix
            tbar.set_description('Epoch %d: Current loss: %4f' % (epoch + 1, loss))
            running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        # Epoch evaluation
        running_loss = running_loss / train_dataloader_len
        if epoch % 1 == 0 or epoch == epochs - 1 or early_stopping_count >= early_stopping_patience:
            with torch.no_grad():
                train_acc = test(model, train_dataloader, device, None)
                test_acc = test(model, test_dataloader, device, None)
                print('Epoch %s: Loss: %s | Test > Acc %s | Train > Acc %s' % (
                str(epoch + 1, ), str(running_loss), str(test_acc.round(decimals=2)), str(train_acc.round(decimals=2))))
                logger.info('Epoch %s: Loss: %s | Test > Acc %s | Train > Acc %s' % (
                str(epoch + 1, ), str(running_loss), str(test_acc.round(decimals=2)), str(train_acc.round(decimals=2))))

                training_acc.append(train_acc)
                testing_acc.append(test_acc)
                training_epoch.append(epoch + 1)

            if test_acc > best_acc:
                torch.save(model.state_dict(), model_save_file)
                best_acc = test_acc
                early_stopping_count = 0
            else:
                early_stopping_count += 1
                print('Early stopping count: %d/%d.' % (early_stopping_count, early_stopping_patience))
                logger.info('Early stopping count: %d/%d.' % (early_stopping_count, early_stopping_patience))

        training_loss.append(running_loss)

        # Early stopping
        if early_stopping_count >= early_stopping_patience:
            print('Early stopped at epoch %d.' % (epoch + 1))
            logger.info('Early stopped at epoch %d.' % (epoch + 1))
            break

    return training_loss, training_epoch, training_acc, testing_acc


# Test function
def test(model, test_dataloader, device, model_save_file=None):

    # Switch to testing mode
    if model_save_file is not None:
        model.load_state_dict(torch.load(model_save_file))
    model.eval()

    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)

    with torch.no_grad():
        tbar = tqdm(test_dataloader, desc="Valid/Test phase", smoothing=0, mininterval=1.0)
        for i, (x_categ, x_numer, y, x_categ_mask, x_numer_mask) in enumerate(tbar):
            x_categ = x_categ.to(device)
            x_numer = x_numer.to(device)
            x_categ_mask = x_categ_mask.to(device)
            x_numer_mask = x_numer_mask.to(device)
            y = torch.squeeze(y).to(device)

            y_outs = model(x_categ, x_numer, x_categ_mask, x_numer_mask)
            y_outs = torch.argmax(y_outs, dim=1)

            y_test = torch.cat([y_test, y.float()], dim=0)
            y_pred = torch.cat([y_pred, y_outs.float()], dim=0)

        # Instance-level
        correct_results_sum = (y_pred == y_test).float().sum()
        acc = correct_results_sum / y_test.shape[0] * 100

    model.train()

    return acc.cpu().numpy()


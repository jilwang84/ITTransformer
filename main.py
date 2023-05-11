# Copyright (c) 2023-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import os
import logging
from datetime import datetime
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from src.TabularTransformer import TabularTransformer
from src.train_test import train, test
from src.result_saver import Result_Saver, save_training_loss
from src.dataset import TabDataset, data_prep_openml
from src.scheduler import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader


if not os.path.exists('./log'):
    os.mkdir('./log')
if not os.path.exists('./result'):
    os.mkdir('./result')
time_format = "%Y-%b-%d_%H-%M-%S"

# -------------- Deep Tabular Learning Script --------------
if 1:
    # ---- Command Line Arguments Section ------------------
    parser = argparse.ArgumentParser()

    # General config
    parser.add_argument('--model', type=str, default='ITTransformer',
                        choices=['TabTransformer', 'FTTransformer', 'ITTransformer'],
                        help='Available model: TabTransformer, FTTransformer, ITTransformer. Default: ITTransformer')
    parser.add_argument('--batch', type=int, default=256,
                        help='Batch size. Default 256.')
    parser.add_argument('--dataset', type=str, default='income',
                        choices=['adult', 'covertype', 'income', 'bank', 'volkert', 'Diabetes130US'],
                        help='Dataset name: income, bank. Default income.')
    parser.add_argument('--use_cpu', action='store_true', default=False,
                        help='Disables CUDA training and uses CPU only. Default False.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for torch. Default None.')
    parser.add_argument('--dataset_seed', type=int, default=42,
                        help='Random seed for dataset generation. Default 42.')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='The GPU to be used. Default 0.')

    # Optim config
    parser.add_argument('--epoch', type=int, default=100,
                        help='The number of epochs for training. Default 100.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate. Default 0.0.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate. Default 0.0001.')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'Adam', 'RAdam'],
                        help='Use optimizer for training: AdamW, Adam, RAdam. Default AdamW.')
    parser.add_argument('--scheduler', type=str, default='none',
                        choices=['cosine', 'linear', 'cosine_warmup', 'none'],
                        help='Use scheduler for training: cosine, linear, cosine_warmup, none. Default none.')
    parser.add_argument('--early_stop', type=int, default=None,
                        help='Early stopping patience. Default None.')
    parser.add_argument('--loss', type=str, default='CrossEntropy',
                        choices=['CrossEntropy', 'KLDiv'],
                        help='Loss function. Default CrossEntropy.')

    # Command line arguments parser
    print('********** Parsing Parameter ***********')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    import torch
    import torch.optim as optim
    import torch.nn as nn
    args.use_cuda = not args.use_cpu and torch.cuda.is_available()
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)
        print('CUDA:', args.use_cuda, ' Seed:', args.seed)
    else:
        print('CUDA:', args.use_cuda)

    # Initialize the logger
    main_title = str(args.gpu_id) + '_' + args.model + '_' + args.dataset + '_b' + str(args.batch) + '_' + args.loss
    train_title = 'e' + str(args.epoch) + '_dp' + str(args.dropout) + '_es' + str(args.early_stop) + '_lr' + str(
        args.lr) + '_' + str(args.optimizer)
    title = main_title + '_#_' + train_title

    model_save_path = os.path.join(os.getcwd(), 'result', title)
    logging_file_path = os.path.join(os.getcwd(), 'log', title)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(logging_file_path, exist_ok=True)

    logging_file_name = os.path.join(logging_file_path, datetime.now().strftime(time_format) + '.log')
    logging.basicConfig(filename=logging_file_name, format='[%(asctime)s][%(levelname)s] - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S %p", level=logging.INFO)
    logger = logging.getLogger()

    logger.info("Arguments: " + str(args))
    # ------------------------------------------------------


def main():

    # ---- Objection Initialization Section ----------------
    print('************ Initialization ************')
    logger.info('************ Initialization ************')

    # Loading dataset
    if args.dataset == 'adult':
        openml_id = 1590
    elif args.dataset == 'covertype':
        openml_id = 1596
    elif args.dataset == 'income':
        openml_id = 4535
    elif args.dataset == 'bank':
        openml_id = 44234
    elif args.dataset == 'volkert':
        openml_id = 41166
    elif args.dataset == 'Diabetes130US':
        openml_id = 4541
    else:
        print('The given dataset is not acceptable.')
        logger.info('The given dataset is not acceptable.')
        raise NotImplementedError

    cat_dims, cat_idx, con_idx, X_train, y_train, X_test, y_test, train_mean, train_std, n_class = data_prep_openml(
        openml_id, args.dataset_seed)
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)
    train_dataset = TabDataset(X_train, y_train, cat_idx, con_idx, continuous_mean_std)
    test_dataset = TabDataset(X_test, y_test, cat_idx, con_idx, continuous_mean_std)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)
    print('Loading data done. The target of %s dataset has %d classes.' % (args.dataset, n_class))
    logger.info('Loading data done. The target of %s dataset has %d classes.' % (args.dataset, n_class))

    # Prepare model saver
    result_saver = Result_Saver(args.model + ' Model Saver', 'Model Parameters')
    model_save_file = os.path.join(model_save_path, datetime.now().strftime(time_format) + '.pth')
    result_saver.result_destination_file_path = model_save_file

    # ---- Parameter Section -------------------------------
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

    # Usage:
    if args.model == 'TabTransformer':
        # TabTransformer hyperparameters
        embedding_size = 32
        transformer_depth = 6
        attention_heads = 8
        dim_head = 16
        mlp_hidden_mults = (4, 2)
        mlp_act = nn.ReLU()
        attention_dropout = args.dropout
        ff_dropout = args.dropout
        cont_embeddings = 'None'
        attention_type = 'col'
        cls_only = False

        model = TabularTransformer(
            categories=tuple(cat_dims),
            num_continuous=len(con_idx),
            dim=embedding_size,
            depth=transformer_depth,
            heads=attention_heads,
            dim_head=dim_head,
            dim_out=n_class,
            mlp_hidden_mults=mlp_hidden_mults,
            mlp_act=mlp_act,
            attn_dropout=attention_dropout,
            ff_dropout=ff_dropout,
            cont_embeddings=cont_embeddings,
            attention_type=attention_type,
            cls_only=cls_only
        ).to(device)
    elif args.model == 'FTTransformer':
        # FTTransformer hyperparameters
        embedding_size = 32
        transformer_depth = 6
        attention_heads = 8
        dim_head = 16
        mlp_hidden_mults = (4, 2)
        mlp_act = nn.ReLU()
        attention_dropout = args.dropout
        ff_dropout = args.dropout
        cont_embeddings = 'MLP'
        attention_type = 'col'
        cls_only = True

        model = TabularTransformer(
            categories=tuple(cat_dims),
            num_continuous=len(con_idx),
            dim=embedding_size,
            depth=transformer_depth,
            heads=attention_heads,
            dim_head=dim_head,
            dim_out=n_class,
            mlp_hidden_mults=mlp_hidden_mults,
            mlp_act=mlp_act,
            attn_dropout=attention_dropout,
            ff_dropout=ff_dropout,
            cont_embeddings=cont_embeddings,
            attention_type=attention_type,
            cls_only=cls_only
        ).to(device)
    elif args.model == 'ITTransformer':
        # ITTransformer hyperparameters
        embedding_size = 32
        transformer_depth = 5
        attention_heads = 8
        dim_head = 16
        mlp_hidden_mults = (4, 2)
        mlp_act = nn.ReLU()
        attention_dropout = args.dropout
        ff_dropout = args.dropout
        cont_embeddings = 'MLP'
        attention_type = 'colrow'
        cls_only = True

        model = TabularTransformer(
            categories=tuple(cat_dims),
            num_continuous=len(con_idx),
            dim=embedding_size,
            depth=transformer_depth,
            heads=attention_heads,
            dim_head=dim_head,
            dim_out=n_class,
            mlp_hidden_mults=mlp_hidden_mults,
            mlp_act=mlp_act,
            attn_dropout=attention_dropout,
            ff_dropout=ff_dropout,
            cont_embeddings=cont_embeddings,
            attention_type=attention_type,
            cls_only=cls_only
        ).to(device)
    else:
        print('The given model is not acceptable.')
        logger.info('The given model is not acceptable.')
        raise NotImplementedError
    print('Model initialization done. The model is %s.' % args.model)
    logger.info('Model initialization done. The model is %s.' % args.model)
    # ------------------------------------------------------

    # ---- Training Section --------------------------------
    print('************ Training Start ************')
    logger.info('************ Training Start ************')
    # Training parameters
    if args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    elif args.loss == 'KLDiv':
        criterion = nn.KLDivLoss(reduction="batchmean").to(device)
    else:
        print('The given loss function is not acceptable.')
        logger.info('The given loss function is not acceptable.')
        raise NotImplementedError

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=args.lr)
    else:
        print('The given optimizer is not acceptable.')
        logger.info('The given optimizer is not acceptable.')
        raise NotImplementedError

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch // 2.667, args.epoch // 1.6,
                                                                                args.epoch // 1.142], gamma=0.1)
    elif args.scheduler == 'cosine_warmup':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, args.epoch, max_lr=args.lr, min_lr=1e-9,
                                                  warmup_steps=int(0.1 * args.epoch))
    else:
        scheduler = None

    start = time.time()
    training_loss, training_epoch, train_acc, test_acc = train(model, train_dataloader, test_dataloader, device,
                                                               optimizer, scheduler, args.epoch, criterion,
                                                               args.early_stop, logger, model_save_file)
    end = time.time()
    print('************ Training End **************')
    logger.info('************ Training End **************')
    print("Training run time: %f s" % (end - start))
    logger.info("Training run time: %f s" % (end - start))

    # Training loss plot
    save_training_loss(training_loss,
                       os.path.join(model_save_path, datetime.now().strftime(time_format) + '_training_loss.txt'))

    plt.figure(figsize=(10, 5))
    plt.title("Training Loss")
    plt.plot(training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(model_save_path, datetime.now().strftime(time_format) + '_training_loss.png'))
    plt.figure(figsize=(10, 5))
    plt.title("Training Accuracy")
    plt.plot(training_epoch, train_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(model_save_path, datetime.now().strftime(time_format) + 'train_acc.png'))
    plt.figure(figsize=(10, 5))
    plt.title("Testing Accuracy")
    plt.plot(training_epoch, test_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(model_save_path, datetime.now().strftime(time_format) + 'test_acc.png'))
    # ------------------------------------------------------

    # ---- Testing Section ---------------------------------
    print('************ Testing Start *************')
    logger.info('************ Testing Start *************')
    start = time.time()
    acc = test(model, test_dataloader, device, model_save_file)
    end = time.time()
    if os.path.exists(model_save_file):
        os.remove(model_save_file)
    print('********** Overall Performance *********')
    logger.info('********** Overall Performance *********')
    print('Task summary: ', title)
    logger.info('Task summary: %s' % title)
    print('Accuracy:', str(acc))
    logger.info('Accuracy: %s' % (str(acc)))
    print("Test time: %f s" % (end - start))
    logger.info("Test time: %f s" % (end - start))
    print('**************** Finish ****************')
    logger.info('**************** Finish ****************')

    # Used for automatic experiments analysis
    f = open(str(args.gpu_id) + "_acc.txt", "w")
    f.write(str(acc))
    f.close()
    # ------------------------------------------------------


if __name__ == '__main__':
    main()


import argparse
import sys
import torch
from bimodel.bert_model import BiModel
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import os
from bi_utils.data_processer import BiDataset
from bi_utils import metrics
import time
from tqdm import tqdm
sys.path.append("..")
import config



def run_train(train_dataloader, model, optimizer, loss_fn, args):
    model.train()
    acces = 0
    losses = 0
    F1s = 0
    num_samples = len(train_dataloader.dataset)

    for batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels in tqdm(train_dataloader):
        input_ids, attention_mask, token_type_ids, labels = batch_seqs.to(
            config.device), batch_seq_masks.to(config.device), batch_seq_segments.to(
            config.device), batch_labels.to(config.device)
        optimizer.zero_grad()
        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids
                       )
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        pred = torch.argmax(output,axis=1)
        acc = metrics.compute_acc(pred, labels)
        F1 = metrics.f1(pred, labels)
        acces += acc
        losses += loss
        F1s += F1

    losses *= args.batch_size / num_samples
    acces *= args.batch_size / num_samples
    F1s *= args.batch_size / num_samples

    return losses, acces, F1s


def run_evaluate(dev_dataloader, model, loss_fn, args):
    model.train()
    acces = 0
    losses = 0
    F1s = 0
    num_samples = len(dev_dataloader.dataset)
    with torch.no_grad():
        for batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels in dev_dataloader:
            input_ids, attention_mask, token_type_ids, labels = batch_seqs.to(
                config.device), batch_seq_masks.to(config.device), batch_seq_segments.to(
                config.device), batch_labels.to(config.device)
            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids
                           )
            loss = loss_fn(output, labels)
            pred = torch.argmax(output,axis=1)
            acc = metrics.compute_acc(pred, labels)
            F1 = metrics.f1(pred, labels)
            acces += acc
            losses += loss
            F1s += F1

    losses *= args.batch_size / num_samples
    acces *= args.batch_size / num_samples
    F1s *= args.batch_size / num_samples

    return losses, acces, F1s


def main():
    args = config.get_args()
    model = BiModel(args).to(config.device)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)

    print('\t* Loading training data...')
    train_data = BiDataset(filename=config.bi_train,
                               tokenizer=tokenizer,
                               args=args)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    print('\t* Loading validation data...')
    val_data = BiDataset(filename=config.bi_dev,
                             tokenizer=tokenizer,
                             args=args)
    dev_dataloader = DataLoader(val_data, shuffle=True, batch_size=args.batch_size)

    best_f1 = 0
    bi_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        time1 = time.time()
        train_loss, train_acc, train_f1 = run_train(train_dataloader=train_dataloader,
                                                    model=model,
                                                    optimizer=optimizer,
                                                    loss_fn=bi_loss_fn,
                                                    args=args)
        dev_loss, dev_acc, dev_f1 = run_evaluate(dev_dataloader=dev_dataloader,
                                                 model=model,
                                                 loss_fn=bi_loss_fn,
                                                 args=args)

        print(f"train: f1:{train_f1 * 100:.2f} loss:{train_loss:.3f} acc:{train_acc * 100:.2f}% "
              f"dev: f1:{dev_f1 * 100:.2f} loss:{dev_loss:.3f} acc:{dev_acc * 100:.2f}% "
              f"time {time.time()-time1:.1f} s")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            if not os.path.exists('checkout/state_dict'):
                os.mkdir('checkout/state_dict')
            torch.save(model.state_dict(), config.bi_weight)


if __name__ == '__main__':
    main()

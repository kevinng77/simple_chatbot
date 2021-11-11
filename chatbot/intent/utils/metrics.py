import torch


def f1(pred, target):
    correct = torch.sum((pred == target)*target)
    # print("correct", correct)
    precision = correct / (torch.sum(pred)+1e-9)
    # print("precision",precision)
    recall = correct / (torch.sum(target)+1e-9)
    # print("recall",recall)
    return 2 * (recall * precision) / (recall + precision + 1e-9)


def macro_f1(pred, target):
    a = f1(pred,target)
    b = f1(~pred,1-target)
    return (a+b)/2


def compute_acc(pred, target):
    correct = torch.sum(pred == target)
    return correct / target.size()[0]
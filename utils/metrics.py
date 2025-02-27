import torch
import numpy as np
#from scipy.special import logsumexp

from sklearn.metrics import roc_curve, auc, precision_recall_curve, recall_score, precision_score, confusion_matrix

class Estimator():
    def __init__(self, cfg, thresholds=None):
        self.cfg = cfg
        self.criterion = cfg.train.criterion
        self.num_classes = cfg.data.num_classes
        self.thresholds = [-0.5 + i for i in range(self.num_classes)] if not thresholds else thresholds
        self.bin_thresholds = [-0.5 + i for i in range(2)] if not thresholds else thresholds

        self.reset()  # intitialization

    def update(self, predictions, targets):
        targets = targets.data.cpu()
        bin_targets = (targets >= self.cfg.data.threshold).long()

        predictions = predictions.data.cpu()
        if self.cfg.data.binary:
            self.bin_prediction.append(predictions.detach())
            self.onset_target.append(targets.detach())

        predictions = self.to_prediction(predictions)
        bin_prediction = (predictions >= self.cfg.data.threshold).long()

        # update metrics
        self.num_samples += len(predictions)
        self.correct += (predictions == targets).sum().item()
        self.bin_correct += (bin_prediction == bin_targets).sum().item()
        for i, p in enumerate(predictions):
            self.conf_mat[int(targets[i])][int(p.item())] += 1
        
        for i, p in enumerate(bin_prediction):
            self.bin_conf_mat[int(bin_targets[i])][int(p.item())] += 1

        #print('end')
    
    def get_auc_auprc(self, digits=-1):
        if self.cfg.data.binary:
            y_bin_pred = torch.cat(self.bin_prediction, dim=0) # x axis
            y_onset_target = torch.cat(self.onset_target, dim=0)
            y_pred_proba = torch.nn.functional.softmax(y_bin_pred, 1) # y-axis
            y_pred = torch.argmax(y_pred_proba, dim=1)            

            y_pred = y_pred.numpy()
            y_pred_proba = y_pred_proba.numpy()
            y_onset_target = y_onset_target.numpy()

            cm = confusion_matrix(y_onset_target, y_pred)
            rec_score = round(recall_score(y_onset_target, y_pred), digits)  
            
            #print(f'test len {len(cm)}', cm)
            if len(cm) > 1: # check
                if (cm[0,1] + cm[1,1]) !=0:
                    prec_score = round(precision_score(y_onset_target, y_pred), digits)
                else:
                    prec_score = 0

                if (cm[0,0] + cm[0,1]) !=0:
                    specificity_score = round(cm[0,0]/(cm[0,0] + cm[0,1]), digits)
                else:
                    specificity_score = 0
            else:
                specificity_score = 1
                prec_score = 1                

            fpr, tpr, thres = roc_curve(y_onset_target, y_pred_proba[:, 1]) # pos_label=1 -> when not 0 and 1
            precision, recall, _ = precision_recall_curve(y_onset_target, y_pred_proba[:, 1])
            

            bin_auc = auc(fpr, tpr)
            au_prc = auc(recall, precision)
            
            list_auc = [round(bin_auc, digits), fpr, tpr]
            list_auprc = [round(au_prc, digits), precision, recall]
            list_others = [rec_score, prec_score, specificity_score, cm]
            
            return list_auc, list_auprc, list_others
        else:
            return 0, 0

    def update_val_loss(self, loss):
        self.val_loss.append(loss)

    def get_val_loss(self, digits=-1):
        return round(self.val_loss[-1], digits)

    def get_accuracy(self, digits=-1):
        acc = self.correct / self.num_samples
        bin_acc = self.bin_correct / self.num_samples
        acc = acc if digits == -1 else round(acc, digits)
        bin_acc = bin_acc if digits == -1 else round(bin_acc, digits)
        return acc, bin_acc

    def get_kappa(self, digits=-1):
        kappa = quadratic_weighted_kappa(self.conf_mat)
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa

    def reset(self):
        self.correct = 0
        self.bin_correct =0
        self.num_samples = 0
        self.val_loss = []
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)
        self.bin_conf_mat = np.zeros((2, 2), dtype=int)
        self.bin_prediction = [] # for AUC, ROC, AUPRC
        self.onset_target = []   # for AUC, ROC, AUPRC

    def to_prediction(self, predictions):
        if self.criterion in ['cross_entropy', 'focal_loss', 'kappa_loss']:
            predictions = torch.tensor(
                [torch.argmax(p) for p in predictions]
            ).long()
        elif self.criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:
            predictions = torch.tensor(
                [self.classify(p.item()) for p in predictions]
            ).float()
        else:
            raise NotImplementedError('Not implemented criterion.')

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i


def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()

    return (observed - expected) / (1 - expected)

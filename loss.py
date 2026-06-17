import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

def negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale):

    # print("img_embs mean/std:", img_embs.mean().item(), img_embs.std().item())
    # print("text_embs mean/std:", text_embs.mean().item(), text_embs.std().item())
    # print("neg_text_embs mean/std:", neg_text_embs.mean().item(), neg_text_embs.std().item())

    img_embs = F.normalize(img_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)
    neg_text_embs = F.normalize(neg_text_embs, dim=-1)

    # Normalize embeddings
    batch_size = img_embs.shape[0]
    labels = torch.arange(batch_size, device=img_embs.device).long()
    # print(f"img_embs.shape: {img_embs.shape}")
    # print(f"text_embs.shape: {text_embs.shape}")
    
    img_text_similarity = logit_scale * img_embs @ text_embs.t()
    # print(img_text_similarity)
    # print("img_text_similarity mean/std:", img_text_similarity.mean().item(), img_text_similarity.std().item())
    # print("img_text_similarity min/max:", img_text_similarity.min().item(), img_text_similarity.max().item())
    text_img_similarity = logit_scale * text_embs @ img_embs.t()
    # print("text_img_similarity mean/std:", text_img_similarity.mean().item(), text_img_similarity.std().item())
    # print("text_img_similarity min/max:", text_img_similarity.min().item(), text_img_similarity.max().item())
    img_negtext_similarity = logit_scale * img_embs @ neg_text_embs.t()
    # print("img_negtext_similarity mean/std:", img_negtext_similarity.mean().item(), img_negtext_similarity.std().item())
    # print("img_negtext_similarity min/max:", img_negtext_similarity.min().item(), img_negtext_similarity.max().item())

    preds_i2t = torch.cat((img_text_similarity, img_negtext_similarity), dim=-1).argmax(dim=-1)
    preds_t2i = img_text_similarity.t().argmax(dim=-1)
    acc_i2t = (preds_i2t == labels).float().mean().item()
    acc_t2i = (preds_t2i == labels).float().mean().item()
    accuracy = (acc_i2t + acc_t2i) / 2

    loss = (
        F.cross_entropy(
            torch.cat([img_text_similarity, img_negtext_similarity], dim=-1), labels
        )
        + F.cross_entropy(text_img_similarity, labels)
    ).div(2)

    return loss, accuracy




def tripletclip_loss(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale):
    loss_1, accuracy1 = negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale)
    loss_2, accuracy2 = negclip_loss(neg_img_embs, neg_text_embs, text_embs, logit_scale)
    # loss_2, accuracy2 = negclip_loss(text_embs, img_embs, neg_img_embs, logit_scale)

    loss = loss_1 + loss_2
    print("loss_1:",loss_1)
    print("loss_2:",loss_2)
    accuracy = (accuracy1 + accuracy2) / 2
    # 维度检查
    # print(f"img_embs.shape: {img_embs.shape}")
    # print(f"text_embs.shape: {text_embs.shape}")
    # print(f"neg_img_embs.shape: {neg_img_embs.shape}")
    # print(f"neg_text_embs.shape: {neg_text_embs.shape}")

    return loss, accuracy

def clip_loss(img_embs, text_embs, logit_scale):
    # Normalize embeddings
    batch_size = img_embs.shape[0]
    labels = torch.arange(batch_size, device=img_embs.device).long()

    img_text_similarity = logit_scale * img_embs @ text_embs.t()
    text_img_similarity = logit_scale * text_embs @ img_embs.t()

    preds_i2t = img_text_similarity.argmax(dim=-1)
    preds_t2i = text_img_similarity.argmax(dim=-1)
    acc_i2t = (preds_i2t == labels).float().mean().item()
    acc_t2i = (preds_t2i == labels).float().mean().item()
    accuracy = (acc_i2t + acc_t2i) / 2

    loss = (
        F.cross_entropy(img_text_similarity, labels)
        + F.cross_entropy(text_img_similarity, labels)
    ).div(2)
    return loss, accuracy

def _confusion_mat(label, pred):
    mat = np.zeros((40, 40))
    for _label, _pred in zip(label, pred):
        mat[_label, _pred] += 1
    return mat

class CE:
    def __init__(self, model):
        self.model = model
        self.ce = nn.CrossEntropyLoss()
        self.ce_pretrain = nn.CrossEntropyLoss(ignore_index=0)

    def computeft(self, batch):
        seqs, labels ,clip,clip_moreinf= batch
        #print(labels)
        lastrep, rep, scores = self.model(seqs)  # B * N
        labels = labels.view(-1).long()
        loss = self.ce(scores, labels)
        return loss

    def compute(self, batch):
        seqs, labels = batch
        #print(labels)
        outputs = self.model(seqs)  # B * N
        labels = labels.view(-1).long()
        loss = self.ce(outputs, labels)
        return loss


    def computefreq(self, batch):
        seqs, labels ,clip,clip_moreinf,neg_image,neg_text= batch
        #print(labels)
        lastrep,attn_encoded,scores = self.model(seqs)  # B * N
        labels = labels.view(-1).long()
        loss = self.ce(scores, labels)
        return loss

class Align:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='mean')  # 定义均方误差损失函数
        self.ce = nn.CrossEntropyLoss()

    def compute(self, rep_mask, rep_mask_prediction):
        align_loss = self.mse(rep_mask, rep_mask_prediction)
        return align_loss

class CM:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='mean')
        self.cos = nn.CosineEmbeddingLoss()  # 定义余弦嵌入损失

    def compute(self, clip_pred, clip):
        target_labels = torch.ones(len(clip_pred)).to(clip_pred.device)  # 目标相似度标签
        cosine_loss=self.cos(clip_pred, clip,target_labels)
        return cosine_loss

# class CM:
#     def __init__(self):
#         self.mse = nn.MSELoss(reduction='mean')
#         self.cos = nn.CosineEmbeddingLoss()

#     def compute(self, img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale):
#         loss, accuracy = tripletclip_loss(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale)
#         return loss, accuracy

# class CM(nn.Module):
#     def __init__(self):
#         super(CM, self).__init__()
#         self.mse = nn.MSELoss(reduction='mean')
#         self.cos = nn.CosineEmbeddingLoss()

#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#     def compute(self, img_embs, text_embs, neg_img_embs, neg_text_embs):
#         logit_scale = self.logit_scale.exp().clamp(0, 100)
#         loss, acc = tripletclip_loss(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale)
#         return loss, acc

class Reconstruct:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.2)

    def compute(self, token_prediction_prob, tokens):
        hits = torch.sum(torch.argmax(token_prediction_prob, dim=-1) == tokens)
        NDCG10 = recalls_and_ndcgs_for_ks(token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
                                          tokens.reshape(-1, 1), 10)
        reconstruct_loss = self.ce(token_prediction_prob.view(-1, token_prediction_prob.shape[-1]), tokens.view(-1))
        return reconstruct_loss, hits, NDCG10


def recalls_and_ndcgs_for_ks(scores, answers, k):
    answers = answers.tolist()
    labels = torch.zeros_like(scores).to(scores.device)
    for i in range(len(answers)):
        labels[i][answers[i]] = 1
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    cut = cut[:, :k]
    hits = labels_float.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits * weights.to(hits.device)).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
    ndcg = (dcg / idcg).mean()
    ndcg = ndcg.cpu().item()
    return ndcg



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import copy

# def negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale):
#     # Normalize embeddings
#     batch_size = img_embs.shape[0]
#     labels = torch.arange(batch_size, device=img_embs.device).long()

#     img_text_similarity = logit_scale * img_embs @ text_embs.t()
#     text_img_similarity = logit_scale * text_embs @ img_embs.t()
#     img_negtext_similarity = logit_scale * img_embs @ neg_text_embs.t()

#     preds_i2t = torch.cat((img_text_similarity, img_negtext_similarity), dim=-1).argmax(
#         dim=-1
#     )
#     preds_t2i = img_text_similarity.t().argmax(dim=-1)
#     acc_i2t = (preds_i2t == labels).float().mean().item()
#     acc_t2i = (preds_t2i == labels).float().mean().item()
#     accuracy = (acc_i2t + acc_t2i) / 2

#     loss = (
#         F.cross_entropy(
#             torch.cat([img_text_similarity, img_negtext_similarity], dim=-1), labels
#         )
#         + F.cross_entropy(text_img_similarity, labels)
#     ).div(2)
#     return loss, accuracy




# def tripletclip_loss(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale):
#     loss_1, accuracy1 = negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale)
#     loss_2, accuracy2 = negclip_loss(neg_img_embs, neg_text_embs, text_embs, logit_scale)
#     # loss_2, accuracy2 = negclip_loss(text_embs, img_embs, neg_img_embs, logit_scale)

#     loss = loss_1 + loss_2
#     # print("loss_1:",loss_1)
#     # print("loss_2:",loss_2)
#     accuracy = (accuracy1 + accuracy2) / 2
#     # 维度检查
#     # print(f"img_embs.shape: {img_embs.shape}")
#     # print(f"text_embs.shape: {text_embs.shape}")
#     # print(f"neg_img_embs.shape: {neg_img_embs.shape}")
#     # print(f"neg_text_embs.shape: {neg_text_embs.shape}")

#     return loss, accuracy

# def clip_loss(img_embs, text_embs, logit_scale):
#     # Normalize embeddings
#     batch_size = img_embs.shape[0]
#     labels = torch.arange(batch_size, device=img_embs.device).long()

#     img_text_similarity = logit_scale * img_embs @ text_embs.t()
#     text_img_similarity = logit_scale * text_embs @ img_embs.t()

#     preds_i2t = img_text_similarity.argmax(dim=-1)
#     preds_t2i = text_img_similarity.argmax(dim=-1)
#     acc_i2t = (preds_i2t == labels).float().mean().item()
#     acc_t2i = (preds_t2i == labels).float().mean().item()
#     accuracy = (acc_i2t + acc_t2i) / 2

#     loss = (
#         F.cross_entropy(img_text_similarity, labels)
#         + F.cross_entropy(text_img_similarity, labels)
#     ).div(2)
#     return loss, accuracy

# def _confusion_mat(label, pred):
#     mat = np.zeros((40, 40))
#     for _label, _pred in zip(label, pred):
#         mat[_label, _pred] += 1
#     return mat

# class CE:
#     def __init__(self, model):
#         self.model = model
#         self.ce = nn.CrossEntropyLoss()
#         self.ce_pretrain = nn.CrossEntropyLoss(ignore_index=0)

#     def computeft(self, batch):
#         seqs, labels ,clip,clip_moreinf= batch
#         #print(labels)
#         lastrep, rep, scores = self.model(seqs)  # B * N
#         labels = labels.view(-1).long()
#         loss = self.ce(scores, labels)
#         return loss

#     def compute(self, batch):
#         seqs, labels = batch
#         #print(labels)
#         outputs = self.model(seqs)  # B * N
#         labels = labels.view(-1).long()
#         loss = self.ce(outputs, labels)
#         return loss


#     def computefreq(self, batch):
#         seqs, labels ,clip,clip_moreinf= batch
#         #print(labels)
#         lastrep,attn_encoded,scores = self.model(seqs)  # B * N
#         labels = labels.view(-1).long()
#         loss = self.ce(scores, labels)
#         return loss

# class Align:
#     def __init__(self):
#         self.mse = nn.MSELoss(reduction='mean')
#         self.ce = nn.CrossEntropyLoss()

#     def compute(self, rep_mask, rep_mask_prediction):
#         align_loss = self.mse(rep_mask, rep_mask_prediction)
#         return align_loss

# # class CM:
# #     def __init__(self):
# #         self.mse = nn.MSELoss(reduction='mean')
# #         self.cos = nn.CosineEmbeddingLoss()

# #     def compute(self, clip_pred, clip):
# #         target_labels = torch.ones(len(clip_pred)).to("cuda")  # 目标相似度标签
# #         cosine_loss=self.cos(clip_pred, clip,target_labels)
# #         return cosine_loss

# class CM:
#     def __init__(self):
#         self.mse = nn.MSELoss(reduction='mean')
#         self.cos = nn.CosineEmbeddingLoss()

#     def compute(self, img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale):
#         loss, accuracy = tripletclip_loss(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale)
#         return loss, accuracy
# # class CM(nn.Module):
# #     def __init__(self):
# #         super(CM, self).__init__()
# #         self.mse = nn.MSELoss(reduction='mean')
# #         self.cos = nn.CosineEmbeddingLoss()

# #         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

# #     def compute(self, img_embs, text_embs, neg_img_embs, neg_text_embs):
# #         logit_scale = self.logit_scale.exp().clamp(0, 100)
# #         loss, acc = tripletclip_loss(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale)
# #         return loss, acc

# class Reconstruct:
#     def __init__(self):
#         self.ce = nn.CrossEntropyLoss(label_smoothing=0.2)

#     def compute(self, token_prediction_prob, tokens):
#         hits = torch.sum(torch.argmax(token_prediction_prob, dim=-1) == tokens)
#         NDCG10 = recalls_and_ndcgs_for_ks(token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
#                                           tokens.reshape(-1, 1), 10)
#         reconstruct_loss = self.ce(token_prediction_prob.view(-1, token_prediction_prob.shape[-1]), tokens.view(-1))
#         return reconstruct_loss, hits, NDCG10


# def recalls_and_ndcgs_for_ks(scores, answers, k):
#     answers = answers.tolist()
#     labels = torch.zeros_like(scores).to(scores.device)
#     for i in range(len(answers)):
#         labels[i][answers[i]] = 1
#     answer_count = labels.sum(1)

#     labels_float = labels.float()
#     rank = (-scores).argsort(dim=1)
#     cut = rank
#     cut = cut[:, :k]
#     hits = labels_float.gather(1, cut)
#     position = torch.arange(2, 2 + k)
#     weights = 1 / torch.log2(position.float())
#     dcg = (hits * weights.to(hits.device)).sum(1)
#     idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
#     ndcg = (dcg / idcg).mean()
#     ndcg = ndcg.cpu().item()
#     return ndcg
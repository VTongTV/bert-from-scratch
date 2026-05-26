from torch.utils.data import DataLoader

from data.pretrain_data import PretrainDataset
from data.collator import pretrain_collate


def create_pretrain_dataloader(corpus, tokenizer, batch_size=256, max_len=128, mlm_prob=0.15, shuffle=True):
    dataset = PretrainDataset(corpus, tokenizer, max_len, mlm_prob)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pretrain_collate)

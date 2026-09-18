from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler


def get_standard_sampler(dataset, batch_size: int, shuffle: bool = True):
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return BatchSampler(sampler, batch_size=batch_size, drop_last=False)

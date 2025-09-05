from typing import List, Iterator, Optional
import torch
from torch.utils.data import Sampler

class DynamicBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices with batch sizes that change according to progress
    through the dataset (percentage-based).
    """
    def __init__(
        self,
        dataset,
        percent_intervals: List[int] = [20, 40, 60, 80, 100],
        batch_sizes: List[int] = [32, 64, 128, 256, 512],
        shuffle: bool = True,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        assert len(percent_intervals) == len(batch_sizes), \
            "percent_intervals and batch_sizes must have the same length"
        assert percent_intervals[-1] == 100, \
            "the last value in percent_intervals must be 100"

        self.n = len(dataset)
        self.percent_intervals = percent_intervals
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def _batch_size_for_progress(self, progress_pct: float) -> int:
        for p, bs in zip(self.percent_intervals, self.batch_sizes):
            if progress_pct <= p:
                return bs
        return self.batch_sizes[-1]

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            if self.generator is None:
                idx = torch.randperm(self.n).tolist()
            else:
                idx = torch.randperm(self.n, generator=self.generator).tolist()
        else:
            idx = list(range(self.n))

        i = 0
        while i < self.n:
            progress_pct = (i / self.n) * 100.0
            bs = self._batch_size_for_progress(progress_pct)
            batch = idx[i : i + bs]
            if len(batch) < bs and self.drop_last:
                break
            yield batch
            i += bs

    def __len__(self) -> int:
        # Conservative estimate needed by PyTorch; use smallest batch size
        min_bs = min(self.batch_sizes)
        if self.drop_last:
            return self.n // min_bs
        else:
            return (self.n + min_bs - 1) // min_bs

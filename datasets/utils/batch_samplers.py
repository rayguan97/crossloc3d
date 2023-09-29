import torch
import random
import math
import copy


class ExpansionBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, cfg, log):
        if hasattr(cfg, 'batch_size_expansion_rate'):
            assert cfg.batch_size <= cfg.max_batch_size
            assert cfg.batch_size_expansion_rate > 1

        self.bs = cfg.batch_size
        self.max_bs = cfg.max_batch_size
        self.bs_er = cfg.batch_size_expansion_rate
        self.threshold = cfg.batch_expansion_threshold
        self.dataset = dataset
        self.batches = []
        self.unused = {idx: True for idx in self.dataset.catalog}
        self.log = log
        self.drop_last = cfg.drop_last

    def __len__(self):
        return math.ceil(len(self.unused) / self.bs)

    def expand_batch_size(self):
        assert self.bs_er is not None
        if self.bs >= self.max_bs:
            return
        old_bs = self.bs
        self.bs = round(self.bs_er * self.bs)
        self.bs = min(self.bs, self.max_bs)
        self.log.info('Expanding batch size from %d to %d ...' %
                      (old_bs, self.bs))

    def state_dict(self):
        return dict(
            batch_size=self.bs,
            max_batch_size=self.max_bs,
            batch_size_expansion_rate=self.bs_er,
            batch_expansion_threshold=self.threshold
        )

    def load_state_dict(self, data):
        self.bs = data['batch_size']
        self.max_bs = data['max_batch_size']
        self.bs_er = data['batch_size_expansion_rate']
        self.threshold = data['batch_expansion_threshold']

    def __iter__(self):
        self.create_batches()
        for batch in self.batches:
            yield batch

    def create_batches(self):
        self.batches = []

        unused_elements_ndx = copy.deepcopy(self.unused)
        current_batch = []

        while True:
            if len(current_batch) >= self.bs or len(unused_elements_ndx) == 0:
                if len(current_batch) >= 4:
                    assert len(current_batch) % 2 == 0, 'Incorrect bach size: {}'.format(
                        len(current_batch))
                    if (not self.drop_last) or (len(current_batch) >= self.bs):
                        self.batches.append(current_batch)
                    current_batch = []
                if len(unused_elements_ndx) == 0:
                    break

            selected_element = random.choice(list(unused_elements_ndx))
            unused_elements_ndx.pop(selected_element)
            positives = self.dataset.get_pos_pairs(selected_element)
            if len(positives) == 0:
                continue

            unused_positives = [
                e for e in positives if e in unused_elements_ndx]
            if len(unused_positives) > 0:
                second_positive = random.choice(unused_positives)
                unused_elements_ndx.pop(second_positive)
            else:
                second_positive = random.choice(positives)

            current_batch += [selected_element, second_positive]

        for batch in self.batches:
            assert len(batch) % 2 == 0, 'Incorrect bach size: {}'.format(
                len(batch))

    def update(self, loss):
        rnz = loss['num_non_zero_triplets'] / loss['num_triplets']
        if rnz < self.threshold:
            self.expand_batch_size()

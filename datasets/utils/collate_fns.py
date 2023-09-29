import torch
import MinkowskiEngine as ME
import numpy as np
import time 
def create_collate_fn(dataset, quantization_size=None, ndt=None, create_mask=False):

    def collate_fn(batch):
        meta = {}
        data = {}

        for m, d in batch:
            for k, v in m.items():
                if k not in meta:
                    meta[k] = []
                meta[k].append(v)

            for k, v in d.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)

        for k, v in data.items():
            data[k] = torch.stack(v, 0)

        if dataset.batch_transform is not None:
            # Apply the same transformation on all dataset elements
            data = dataset.batch_transform(data)

        if create_mask:
            positives_mask = [[dataset.catalog[label]['positives'][e]
                               for e in meta['idx']] for label in meta['idx']]
            negatives_mask = [[dataset.catalog[label]['negatives'][e]
                               for e in meta['idx']] for label in meta['idx']]

            positives_mask = torch.tensor(positives_mask)
            negatives_mask = torch.tensor(negatives_mask)

            data['pos_mask'] = positives_mask
            data['neg_mask'] = negatives_mask

        if quantization_size is not None:
            if isinstance(quantization_size, list):
                for k in list(data.keys()):
                    if not k.endswith('pcd'):
                        continue

                    for i, q_size in enumerate(quantization_size):
                        if ndt and ndt[i]:
                            coords = []
                            feats = []
                            for e in data[k]:
                                coord, _ , i = ME.utils.sparse_quantize(coordinates=e, return_index=True, return_inverse=True, quantization_size=q_size)

                                bin = [list() for _ in range(len(coord))]
                                for j in range(len(i)):
                                    bin[i[j]].append(np.array(e[j]))
                                feat = torch.Tensor([
                                    np.concatenate((np.mean(np.array(b), axis=0), np.std(np.array(b), axis=0))) for b in bin
                                    ])
                                coords.append(coord)
                                feats.append(feat)

                            data[k+'_coords_'+str(q_size)] = coords

                            data[k+'_feats_'+str(q_size)] = feats                        
                        else:
                            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=q_size)
                                    for e in data[k]]
                            data[k+'_coords_'+str(q_size)] = coords

                            feats = [torch.ones((coord.shape[0], 1),
                                                dtype=torch.float32) for coord in coords]
                            data[k+'_feats_'+str(q_size)] = feats

                    # t2 = time.time()


                    pcds = [e for e in data[k]]
                    del data[k]
                    data[k] = pcds
                    # print(t2 - t1)

            else:
                for k in list(data.keys()):
                    if not k.endswith('pcd'):
                        continue

                    if ndt:

                        coords = []
                        feats = []
                        for e in data[k]:
                            coord, _ , i = ME.utils.sparse_quantize(coordinates=e, return_index=True, return_inverse=True, quantization_size=quantization_size)

                            bin = [list() for _ in range(len(coord))]
                            for j in range(len(i)):
                                bin[i[j]].append(np.array(e[j]))
                            feat = np.array([
                                np.concatenate((np.mean(np.array(b), axis=0), np.std(np.array(b), axis=0))) for b in bin
                                ])

                            coords.append(coord)
                            feats.append(feat)

                        data[k+'_coords'] = coords

                        data[k+'_feats'] = feats   

                    else:
                        coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=quantization_size)
                                for e in data[k]]
                        # coords = ME.utils.batched_coordinates(coords)
                        # feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                        feats = [torch.ones((coord.shape[0], 1),
                                            dtype=torch.float32) for coord in coords]
                        data[k+'_coords'] = coords
                        data[k+'_feats'] = feats
                    pcds = [e for e in data[k]]
                    del data[k]
                    data[k] = pcds
        return meta, data

    return collate_fn

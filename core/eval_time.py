
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle as pkl
import os.path as osp
from sklearn.neighbors import KDTree
import time
from utils import AverageValue, Metrics
    

def get_recall(m, n, database_vectors, query_vectors, query_sets, num_neighbors=25, db_set=None):
    # Original PointNetVLAD code

    if db_set == None:
        pair_dist = None
    else:
        db_set = db_set[m]
        pair_dist = []

    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    # num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        # {'query': path, 'northing': , 'easting': }
        dist = []
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors)
        # print('dist:', distances)
        # print('inds:', indices)

        for j in range(len(indices[0])):
            # from IPython import embed;embed()
            dist.append(np.linalg.norm(np.array(db_set[indices[0][j]]) - np.array([query_details["northing"], query_details["easting"]]), ord=1))
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

        pair_dist.append(min(dist))

    if num_evaluated == 0:
        return None, None, None, None

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    # from IPython import embed;embed()
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall, pair_dist


def eval(cfg, log, db_data_loader, q_data_loader, task, neighbor=25):

    metrics = AverageValue(Metrics.names())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    task.eval()

    # build database
    t_min = 100

    all_db_embs = []
    all_db_set = []
    for i in tqdm(range(db_data_loader.dataset.subset_len())):
        db_data_loader.dataset.set_subset(i)

        db_embeddings = []
        db_set = []
        for batch_idx, (meta, data) in enumerate(db_data_loader):

            with torch.no_grad():
                t1 = time.time()
                embeddings = task.step(meta, data)
                t2 = time.time()
                t = t2 - t1
                from IPython import embed;embed()
                t_min = min(t2 - t1, t_min)
            if cfg.eval_cfg.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            db_embeddings.append(embeddings.detach().cpu().numpy())
            # from IPython import embed;embed()
            for idx in range(len(meta['idx'])):
                db_set.append((meta["northing"][idx], meta["easting"][idx]))
            if cfg.debug:
                break
        db_embeddings = np.concatenate(db_embeddings, axis=0)
        all_db_embs.append(db_embeddings)
        all_db_set.append(np.array(db_set))

    all_q_embs = []
    for i in tqdm(range(q_data_loader.dataset.subset_len())):
        q_data_loader.dataset.set_subset(i)
        q_embeddings = []
        for batch_idx, (meta, data) in enumerate(q_data_loader):

            with torch.no_grad():
                t1 = time.time()
                embeddings = task.step(meta, data)
                t2 = time.time()
                t = min(t2 - t1, t)
            if cfg.eval_cfg.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            q_embeddings.append(embeddings.detach().cpu().numpy())
            if cfg.debug:
                break
        q_embeddings = np.concatenate(q_embeddings, axis=0)
        all_q_embs.append(q_embeddings)
    # iters = [(i, j) for i in range(len(all_db_embs))
    #          for j in range(len(all_q_embs)) if i != j]

    print(t_min)
    from IPython import embed;embed()


    iters = [(i, j) for i in range(len(all_db_embs))
             for j in range(len(all_q_embs)) if i != j]

    similarity = []
    dist = []
    recall = np.zeros(neighbor)
    with tqdm(total=len(iters)) as pbar:
        for i, j in iters:
            pair_recall, pair_similarity, pair_opr, pair_dist = get_recall(
                i, j, all_db_embs, all_q_embs, q_data_loader.dataset.catalog, num_neighbors=neighbor, db_set=all_db_set)
            if pair_recall is None:
                continue
            _metrics = Metrics.get(pair_opr)
            metrics.update(_metrics)
            pbar.update(1)
            recall += np.array(pair_recall)
            for x in pair_similarity:
                similarity.append(x)
            if pair_dist != None:
                for x in pair_dist:
                    dist.append(x)

    avg_recall = recall / len(iters)
    avg_similarity = np.mean(similarity)
    avg_dist = np.mean(dist)
    log.info(
        '====================== EVALUATE RESULTS ======================')
    format_str = '{sample_num:<10} ' + \
        ' '.join(['{%s:<10}' % vn for vn in metrics.value_names])

    title_dict = dict(
        sample_num='Sample'
    )
    title_dict.update({vn: vn for vn in metrics.value_names})

    log.info(format_str.format(**title_dict))

    overall_dict = dict(
        sample_num=len(iters)
    )
    # from IPython import embed;embed()
    overall_dict.update(
        {vn: "%.4f" % metrics.avg(vn) for vn in metrics.value_names})

    log.info(format_str.format(**overall_dict))

    t = 'Avg. similarity: {:.4f} Avg. dist: {:.4f} Avg. recall @N:\n'+str(avg_recall)
    log.info(t.format(avg_similarity, avg_dist))

    return Metrics('Recall@1%', metrics.avg())

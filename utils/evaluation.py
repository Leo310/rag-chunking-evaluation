from typing import Dict, List, TypedDict
from math import log2


class GoldenTestset(TypedDict):
    question: str
    source: str
    ground_truth_chunks: Dict[str, float]  # chunk_id: score
    ground_truth_answer: str


def calculate_metrics(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    ground_truth_relevancies: List[float] = None,
    K=10,
):
    K = min(K, len(retrieved_chunks))
    if K == 0:
        return {"precision": 0, "recall": 0, "ap": 0, "ndcg": 0}

    average_precision_sum = 0
    hit_count = 0
    dcg = 0
    ground_truth_set = set(ground_truth_chunks)
    for k, retrieved_chunk in enumerate(retrieved_chunks[:K], 1):
        # matched_chunk = None # doing it this way becasue ground truth can be sub-chunk (fact in MultiHopRetrival) of retrieved chunk
        # for chunk in ground_truth_chunks:
        #     if chunk in retrieved_chunk:
        #         matched_chunk = chunk
        #         break
        # if matched_chunk:
        #     hit_count += 1
        #     average_precision_sum += hit_count / k

        #     if ground_truth_relevancies is not None:
        #         rel_k = ground_truth_relevancies[ground_truth_chunks.index(matched_chunk)]
        #         dcg += rel_k / log2(1 + k)
        if retrieved_chunk in ground_truth_set:
            hit_count += 1
            average_precision_sum += hit_count / k
            if ground_truth_relevancies is not None:
                rel_k = ground_truth_relevancies[
                    ground_truth_chunks.index(retrieved_chunk)
                ]
                dcg += rel_k / log2(1 + k)

    precision = hit_count / K
    recall = hit_count / len(ground_truth_chunks) if ground_truth_chunks else 0
    average_precision = (
        average_precision_sum / len(ground_truth_chunks) if ground_truth_chunks else 0
    )

    ndcg = 0
    if ground_truth_relevancies is not None:
        idcg = sum(
            i_rel / log2(1 + k)
            for k, i_rel in enumerate(
                sorted(ground_truth_relevancies, reverse=True)[:K], 1
            )
        )
        ndcg = dcg / idcg if idcg > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "ap": average_precision,
        "ndcg": ndcg,
    }


def calculate_mean_metrics(metrics: List[Dict[str, float]]):
    mean_precision = sum(m["precision"] for m in metrics) / len(metrics)
    mean_recall = sum(m["recall"] for m in metrics) / len(metrics)
    mean_ap = sum(m["ap"] for m in metrics) / len(metrics)
    mean_ndcg = sum(m["ndcg"] for m in metrics) / len(metrics)

    return {
        "precision": mean_precision,
        "recall": mean_recall,
        "map": mean_ap,
        "ndcg": mean_ndcg,
    }

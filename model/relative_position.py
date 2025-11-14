import torch


def t5_relative_position_bucket(T: int, num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
    """
        整数 T を入力して、
        クエリ位置 T-1 から見たキー 0..T-1 までの距離を
        T5形式のバケット番号に変換して返す。
        例:
            >>> t5_relative_position_bucket_int(6)
            tensor([5, 4, 3, 2, 1, 0])
    """

    # キー位置: 0..T-1
    j = torch.arange(T)
    # 距離: クエリ(T-1)からの距離 (0が一番近い)
    d = (T - 1) - j

    # 以下、オリジナルの t5_relative_position_bucket ロジック
    max_exact = num_buckets // 2
    buckets = torch.full_like(d, fill_value=num_buckets - 1, dtype=torch.long)

    is_small = d < max_exact
    buckets[is_small] = d[is_small]

    if (num_buckets - max_exact) > 0:
        log_ratio = torch.log(d[~is_small].float() / max_exact + 1e-8)
        log_base = torch.log(torch.tensor(max_distance / max_exact))
        log_base = torch.clamp(log_base, min=1e-6)
        buckets[~is_small] = max_exact + torch.clamp(
            (log_ratio / log_base * (num_buckets - max_exact)).to(torch.long),
            max=num_buckets - max_exact - 1
        )

    return buckets


def apply_sequence(data, transforms=()):
    if len(transforms) == 0:
        return data
    else:
        if isinstance(data, dict):
            utt2emb = data
            utt2emb_result = {}
            for (utt, x) in utt2emb.items():
                for tr in transforms:
                    x = tr(x)
                utt2emb_result[utt] = x
            return utt2emb_result
        else:
            X = data
            for tr in transforms:
                X = tr(X)
            return X
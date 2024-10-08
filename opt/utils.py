def get_group(group: list[str]):
    res = []
    for g in group:
        res.append(int(g.split('_')[0]))
    return res
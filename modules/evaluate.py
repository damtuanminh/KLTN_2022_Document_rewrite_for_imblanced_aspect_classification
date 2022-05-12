
def cal_aspect_prf(goldens, predicts, num_of_aspect, verbal=False):
    """

    :param verbal:
    :param num_of_aspect:
    :param list of models.AspectOutput goldens:
    :param list of models.AspectOutput predicts:
    :return:
    """
    tp = [0] * (num_of_aspect)
    fp = [0] * (num_of_aspect)
    fn = [0] * (num_of_aspect)

    for g, p in zip(goldens, predicts):
        for i in range(num_of_aspect):
            if g.scores[i] == p.scores[i] == 1:
                tp[i] += 1
            elif g.scores[i] == 1:
                fn[i] += 1
            elif p.scores[i] == 1:
                fp[i] += 1

    p = [tp[i]/(tp[i]+fp[i]) if tp[i] != 0 else 0 for i in range(num_of_aspect)]
    r = [tp[i]/(tp[i]+fn[i]) if tp[i] != 0 else 0 for i in range(num_of_aspect)]
    f1 = [2*p[i]*r[i]/(p[i]+r[i]) if p[i] != 0 else 0 for i in range(num_of_aspect)]

    tpp = [tp[i] for i in range(0, num_of_aspect - 1)]
    fpp = [fp[i] for i in range(0, num_of_aspect - 1)]
    fnn = [fn[i] for i in range(0, num_of_aspect - 1)]
    micro_p = sum(tpp)/(sum(tpp)+sum(fpp))
    micro_r = sum(tpp)/(sum(tpp)+sum(fnn))
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)

    pp = [p[i] for i in range(0, num_of_aspect - 1)]
    rp = [r[i] for i in range(0, num_of_aspect - 1)]
    f1p = [f1[i] for i in range(0, num_of_aspect - 1)]
    macro_p = sum(pp)/(num_of_aspect - 1)
    macro_r = sum(rp)/(num_of_aspect - 1)
    macro_f1 = sum(f1p)/(num_of_aspect - 1)

    if verbal:
        print('p:', p)
        print('r:', r)
        print('f1:', f1)
        print('micro:', (micro_p, micro_r, micro_f1))
        print('macro:', (macro_p, macro_r, macro_f1))

    return p, r, f1, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)

# usage: score.py REF HYP_A [HYP_B]  -> BLEU/chrF++ per system, and A vs B agreement
import sys, sacrebleu, statistics
ref = open(sys.argv[1]).read().rstrip('\n').split('\n')
hyps = [open(p).read().rstrip('\n').split('\n') for p in sys.argv[2:]]
for p, h in zip(sys.argv[2:], hyps):
    r = ref[:len(h)]
    bleu = sacrebleu.corpus_bleu(h, [r]); chrf = sacrebleu.corpus_chrf(h, [r], word_order=2)
    ms = [float(x) for x in open(p + '.ms').read().split()]
    print(f'{p.split("/")[-1]:28s} n={len(h)} BLEU={bleu.score:.2f} chrF++={chrf.score:.2f} mean={statistics.mean(ms):.0f}ms median={statistics.median(ms):.0f}ms p90={sorted(ms)[int(.9*len(ms))]:.0f}ms')
if len(hyps) == 2:
    a, b = hyps; n = min(len(a), len(b))
    same = sum(a[i] == b[i] for i in range(n))
    print(f'identical outputs: {same}/{n}')
    for i in range(n):
        if a[i] != b[i]:
            print(f'  first diff line {i+1}:\n    A: {a[i]}\n    B: {b[i]}'); break

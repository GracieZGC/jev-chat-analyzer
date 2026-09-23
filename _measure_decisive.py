# -*- coding: utf-8 -*-
"""量两件事（都不需要标准答案）：
1. 分差 margin：一级大类 vs 二级标签 vs 意图，哪层是"确定"的、哪层在猜。
2. 大类分布：如果「平静中性」占比过高，说明系统在大量场合退成"没情绪"——
   这正是她连续两次报的问题（暧昧 4 句里 3 句「看不出情绪」）。
"""
import importlib.util, json, re, sys
from collections import defaultdict

spec = importlib.util.spec_from_file_location('srv', 'server.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

raw = open('四场景对话样本_40条.txt', encoding='utf-8').read().split('\n')
segs, cur_seg, cur_rel = [], None, None
for l in raw:
    s = l.strip()
    if s.startswith('===================='):
        cur_rel = next((x for x in ('恋爱', '暧昧', '同事', '上下级') if x in s), cur_rel)
        continue
    mm = re.match(r'^#\s*([LACB])(\d+)', s)
    if mm:
        cur_seg = []
        segs.append((cur_rel, mm.group(0), cur_seg))
        continue
    if s and not s.startswith('#') and cur_seg is not None:
        cur_seg.append(s)

conv = []
for rel, tag, body in segs:
    j = 0
    while j + 2 < len(body):
        if re.match(r'^\d{4}年', body[j + 1]):
            conv.append((rel, tag, body[j], body[j + 2]))
            j += 3
        else:
            j += 1
print('段落数:', len(segs), '| 消息数:', len(conv))
print('按场景:', dict(defaultdict(int, {r: sum(1 for x in conv if x[0] == r) for r in set(x[0] for x in conv)})))

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rows = []
for rel, tag, conv_msgs in [(r, t, [x for x in conv if x[1] == t and x[0] == r]) for r, t, _ in
                            [(x[0], x[1], None) for x in conv]]:
    pass
# 按段跑，保留对话上下文
by_seg = defaultdict(list)
for rel, tag, who, text in conv:
    by_seg[tag].append((rel, who, text))
for tag, lst in by_seg.items():
    rel = lst[0][0]
    for k, (_, who, text) in enumerate(lst[:LIMIT]):
        ctx = '\n'.join(w + '：' + t for _, w, t in lst[max(0, k - 5):k])
        try:
            r = m.evaluate_state({'message': text, 'context': ctx,
                                  'relationship': rel, 'speaker': 'other'}, skip_gen=True)
        except Exception as e:
            print('跳过', type(e).__name__, str(e)[:60])
            continue
        f, e2, pi = r['emotion_family'], r['emotion'], r['primary_intent']
        fr, er, pr = f['ranked'], e2['ranked'], pi['ranked']
        rows.append({'rel': rel, 'seg': tag, 'text': text,
                     'fam': f['key'], 'fs': f['score'],
                     'fgap': round(fr[0]['score'] - (fr[1]['score'] if len(fr) > 1 else 0), 2),
                     'emo': e2['key'], 'es': e2['score'], 'disp': e2['display'],
                     'egap': round(er[0]['score'] - (er[1]['score'] if len(er) > 1 else 0), 2),
                     'coarse': e2.get('coarse'), 'ncand': len(e2['probabilities']),
                     'int': pi['key'], 'is': pi['score'],
                     'igap': round(pr[0]['score'] - (pr[1]['score'] if len(pr) > 1 else 0), 2)})
json.dump(rows, open('/tmp/decisive.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('有效样本:', len(rows))

n = len(rows)


def stat(gapkey, label):
    g = sorted(x[gapkey] for x in rows)
    dec = sum(1 for x in g if x >= 0.15)
    weak = sum(1 for x in g if x < 0.05)
    print(f'{label:8s} 中位分差 {g[n//2]:.2f} | 平均 {sum(g)/n:.2f} '
          f'| 确定(>=.15) {dec}/{n}={dec/n:.0%} | 在猜(<.05) {weak}/{n}={weak/n:.0%}')


print()
stat('fgap', '一级大类')
stat('egap', '二级标签')
stat('igap', '意图层')
print()
cnt = defaultdict(int)
for x in rows:
    cnt[x['fam']] += 1
print('大类分布:', dict(sorted(cnt.items(), key=lambda kv: -kv[1])))
neu = cnt.get('平静中性', 0)
print(f'>>> 「平静中性」占比 {neu}/{n} = {neu/n:.0%}')
ecnt = defaultdict(int)
for x in rows:
    ecnt[x['emo']] += 1
print('二级标签 top8:', dict(sorted(ecnt.items(), key=lambda kv: -kv[1])[:8]))
print('触发 coarse:', sum(1 for x in rows if x['coarse']))
print()
print('=== 判成平静中性的样本（人工看是否真的没情绪）===')
for x in rows:
    if x['fam'] == '平静中性':
        print(f'  [{x["rel"]}] {x["text"][:34]:36s} -> {x["emo"]}({x["es"]}) 展示「{x["disp"]}」')

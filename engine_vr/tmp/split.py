import json, collections
m = json.load(open('engine_vr/manifest.json'))
c = collections.Counter()
for e in m['programs']:
    for b in e['bindings']:
        c[(b['kind'], bool(b.get('explicit_binding')))] += 1
print(dict(c))
print('total', sum(c.values()), 'injected', sum(v for k,v in c.items() if not k[1]), 'explicit', sum(v for k,v in c.items() if k[1]))

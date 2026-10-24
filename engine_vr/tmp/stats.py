import json, collections
m = json.load(open('engine_vr/manifest.json'))
ps = m['programs']
tot = collections.Counter()
for e in ps:
    tot.update(e['port_stats'])
print('programs', len(ps), 'declared bindings', sum(len(e['bindings']) for e in ps))
print('aggregate port stats:', dict(tot))
print('kinds:', dict(collections.Counter(b['kind'] for e in ps for b in e['bindings'])))
print('sets:', dict(collections.Counter(b['set'] for e in ps for b in e['bindings'])))
print('explicit:', sum(1 for e in ps for b in e['bindings'] if b.get('explicit_binding')))
print('max ubo bytes', max((e['uniform_block']['size_bytes'] for e in ps if 'uniform_block' in e), default=0))
print('with explicit:', [e['name'] for e in ps if e['port_stats'].get('explicit_bindings')])
print('outputs:', [(e['name'], [(o['name'], o['location']) for o in e.get('outputs', [])]) for e in ps if len(e.get('outputs', [])) > 1])
print('extras:', json.dumps(m.get('extra_variants_checked')))
print('loose>0?', [(e['name'], e['port_stats']['loose_uniforms']) for e in ps])

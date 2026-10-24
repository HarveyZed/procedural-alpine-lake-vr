import json
m = json.load(open('engine_vr/manifest.json'))
rows = []
for e in m['programs']:
    b = e['bindings']
    s = sum(1 for x in b if x['kind'] == 'combined_sampler')
    im = sum(1 for x in b if x['kind'] == 'storage_image')
    ex = sum(1 for x in b if x.get('explicit_binding'))
    outs = ','.join(o['name'] + '@' + str(o['location']) for o in e.get('outputs', []))
    ub = e.get('uniform_block', {}).get('size_bytes', 0)
    d = ' '.join(k + ('=' + v if v else '') for k, v in e['defines'].items()) or '-'
    rows.append('| ' + e['name'] + ' | ' + e['stage'] + ' | ' + d + ' | ' + str(len(b)) + ' | ' + str(s) +
                ' | ' + str(im) + ' | ' + str(ex) + ' | ' + str(ub) + ' | ' + outs + ' | ' + e['mode'] + ' |')
print('| program | stage | defines | bind | smp | img | expl | UBO B | outputs | mode |')
print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
print('\n'.join(rows))
t = [e for e in m['programs'] if e['name'] in ('scene_trace','scene_cone','scene_kind4_water')]
for e in t:
    print(e['name'], [(x['name'], x['set'], x['binding'], x['glsl_type']) for x in e['bindings']
                      if any(k in x['name'] for k in ('Cone','GBuf','uW2K','uStream'))])

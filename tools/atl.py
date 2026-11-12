import json,numpy as np,os
ROOT='/tmp/src/human-atlas/public/models'
A=json.load(open(ROOT+'/atlas.json'))
_buf={}
def buf(i):
    if i not in _buf: _buf[i]=open(f'{ROOT}/body-{i}.bin','rb').read()
    return _buf[i]
def mesh(p):
    b=buf(p['chunk'])
    v=np.frombuffer(b,dtype='<f4',count=p['vertexCount']*3,offset=p['positions']).reshape(-1,3).astype(np.float64)
    ix=np.frombuffer(b,dtype='<u4',count=p['indexCount'],offset=p['indices'])
    return v,ix
BY={}
for p in A['parts']: BY.setdefault(p['name'].lower(),[]).append(p)
def V(name):
    ps=BY[name.lower()]
    return np.vstack([mesh(p)[0] for p in ps])
def bb(name):
    v=V(name); return v.min(0),v.max(0)
def c(name):
    lo,hi=bb(name); return (lo+hi)/2

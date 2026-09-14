#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('kit',type=Path);a=p.parse_args();path=a.kit/'manifest.json';m=json.loads(path.read_text());old={x['path']:x for x in m['files']};items=[]
    for file in sorted(x for x in a.kit.rglob('*') if x.is_file() and x.name!='manifest.json'):
        rel=str(file.relative_to(a.kit)).replace('\\','/');prior=old.get(rel,{})
        items.append({'path':rel,'bytes':file.stat().st_size,'sha256':sha(file),'producer':prior.get('producer','post-build QA'),'source_table':prior.get('source_table'),'scientific_role':prior.get('scientific_role','supporting release artifact')})
    m['files']=items;path.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8');print(len(items))

if __name__=='__main__':main()

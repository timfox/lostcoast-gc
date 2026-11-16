#!/usr/bin/env python3
"""Append and redirect a loose ISO9660 asset in a separate GameCube diagnostic image.

Patches primary/Joliet directory entries and the disc FST; it does not patch
copies inside ZIP/PK3 archives. The source image and asset are never modified.
"""
import argparse
import hashlib
import struct
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--iso',type=Path,required=True)
p.add_argument('--asset',type=Path,required=True)
p.add_argument('--path',required=True,help='absolute ISO path, e.g. /xash3d/lostcoast/media/foo.gct')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.resolve() in (a.iso.resolve(),a.asset.resolve()):raise SystemExit('output must be separate from input files')
d=bytearray(a.iso.read_bytes()); parts=[x for x in a.path.split('/') if x]
if not parts:raise SystemExit('empty asset path')
records=[]
for sec in range(16,64):
 desc=sec*2048
 if d[desc+1:desc+6]!=b'CD001':raise SystemExit('invalid ISO volume descriptor')
 kind=d[desc]
 if kind==255:break
 if kind not in (1,2):continue
 joliet=kind==2 and d[desc+88:desc+90]==b'%/'
 if kind==2 and not joliet:continue
 rec=desc+156
 for part in parts:
  extent=struct.unpack_from('<I',d,rec+2)[0];size=struct.unpack_from('<I',d,rec+10)[0]
  pos=extent*2048;end=pos+size;found=None
  if end>len(d):raise SystemExit('directory outside image')
  while pos<end:
   length=d[pos]
   if not length:pos=(pos//2048+1)*2048;continue
   if pos+length>end:raise SystemExit('invalid directory record')
   n=d[pos+32];raw=bytes(d[pos+33:pos+33+n])
   try:name=raw.decode('utf-16-be' if joliet else 'ascii').split(';',1)[0]
   except UnicodeError:name=''
   if name.casefold()==part.casefold():found=pos;break
   pos+=length
  if found is None:raise SystemExit(f'asset path missing in descriptor {sec}: {part}')
  rec=found
 records.append(rec)
if not records:raise SystemExit('no ISO filesystem entries found')
fst_off,fst_size=struct.unpack_from('>II',d,0x424);count=struct.unpack_from('>I',d,fst_off+8)[0];names=fst_off+count*12;found=[]
for i in range(1,count):
 word=struct.unpack_from('>I',d,fst_off+i*12)[0]
 if word>>24:continue
 start=names+(word&0xffffff);end=d.find(b'\0',start,fst_off+fst_size)
 if end>=0 and bytes(d[start:end]).decode('utf-8','replace').casefold()==parts[-1].casefold():found.append(fst_off+i*12)
if len(found)!=1:raise SystemExit(f'expected one matching FST asset, got {len(found)}')
d.extend(bytes((-len(d))%2048)); offset=len(d); data=a.asset.read_bytes();d.extend(data);d.extend(bytes((-len(d))%2048))
for rec in records:
 struct.pack_into('<I',d,rec+2,offset//2048);struct.pack_into('>I',d,rec+6,offset//2048)
 struct.pack_into('<I',d,rec+10,len(data));struct.pack_into('>I',d,rec+14,len(data))
for sec in range(16,64):
 desc=sec*2048;kind=d[desc]
 if kind==255:break
 if kind in (1,2):
  struct.pack_into('<I',d,desc+80,len(d)//2048);struct.pack_into('>I',d,desc+84,len(d)//2048)
struct.pack_into('>II',d,found[0]+4,offset,len(data))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(d)
print('asset SHA-256',hashlib.sha256(data).hexdigest(),'ISO SHA-256',hashlib.sha256(d).hexdigest(),'offset',hex(offset),'bytes',len(data))

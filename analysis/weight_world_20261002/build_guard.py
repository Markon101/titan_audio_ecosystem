#!/usr/bin/env python3
"""Preserve the prior binary and build one job with a measured RAM floor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def available():
    return next(int(line.split()[1])/1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def tree_rss(pid):
    output=subprocess.run(['ps','-eo','pid,ppid,rss'],capture_output=True,text=True,check=True).stdout
    rows=[tuple(map(int,line.split())) for line in output.splitlines()[1:] if len(line.split())==3]
    selected={pid}
    while True:
        expanded=selected|{child for child,parent,_ in rows if parent in selected}
        if expanded==selected:break
        selected=expanded
    return sum(rss/1024 for child,_,rss in rows if child in selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='build')
    parser.add_argument('--no-lto',action='store_true')
    parser.add_argument('--light',action='store_true',help='optimize final crate at level 1 with smaller codegen units; keep cached dependencies')
    parser.add_argument('--thin',action='store_true',help='use Cargo-consistent ThinLTO')
    parser.add_argument('--tests',action='store_true',help='run Rust tests sequentially with the same memory guard')
    args=parser.parse_args()
    if not args.name.replace('_','').isalnum():raise ValueError('invalid build name')
    runs=HERE/'runs';runs.mkdir(exist_ok=True)
    prior=runs/'prior_titan'
    if not prior.exists():shutil.copy2(ROOT/'target/release/titan',prior)
    initial=available()
    if initial<2048:raise RuntimeError(f'build postponed: {initial:.0f} MiB available')
    log=runs/f'{args.name}.log'
    if log.exists():raise FileExistsError('preserve prior build log')
    with log.open('xb') as stream:
        command=['cargo','rustc','--release','--locked','--offline','-j1','--','-C','lto=off','-C','codegen-units=1'] if args.no_lto else ['cargo','build','--release','--locked','--offline','-j1']
        if args.light:command=['cargo','rustc','--release','--locked','--offline','-j1','--','-C','lto=off','-C','opt-level=1','-C','codegen-units=16']
        build_env=os.environ.copy()
        if args.thin:
            command=['cargo','build','--release','--locked','--offline','-j1']
            build_env['CARGO_PROFILE_RELEASE_LTO']='thin'
        if args.tests:command=['cargo','test','--locked','--offline','-j1','--','--test-threads=1']
        process=subprocess.Popen(command,cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=build_env)
        peak=0;minimum=initial;stopped=False;start=time.monotonic()
        while process.poll() is None:
            current=available();minimum=min(minimum,current);peak=max(peak,tree_rss(process.pid))
            if current<1024 and not stopped:
                stopped=True;os.killpg(process.pid,signal.SIGTERM)
            time.sleep(0.5)
    result={'schema':1,'command':command,'exit_code':process.returncode,'safety_stop':stopped,
            'cargo_lto_override':'thin' if args.thin else None,
            'elapsed_seconds':time.monotonic()-start,'initial_available_mib':initial,
            'minimum_available_mib':minimum,'peak_tree_rss_mib':peak,
            'prior_binary_sha256':hashlib.sha256(prior.read_bytes()).hexdigest()}
    if process.returncode==0:result['binary_sha256']=hashlib.sha256((ROOT/'target/release/titan').read_bytes()).hexdigest()
    (HERE/f'{args.name}_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if process.returncode or stopped:raise SystemExit(2)


if __name__=='__main__':main()

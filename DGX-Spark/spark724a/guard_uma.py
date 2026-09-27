"""Linux supervisor: one child group, fail closed on low RAM or monitor error.
This is an early-stop watchdog, not a kernel-enforced OOM guarantee.
"""
import os, sys, json, time, signal, subprocess, ctypes, argparse, fcntl
from pathlib import Path
GIB=1024**3

def kv(path):
    return {p[0]: int(p[1]) for line in Path(path).read_text().splitlines() if len(p:=line.split())==2}

def cgroup_dir():
    """First cgroup dir that actually carries memory accounting, else None.
    France: the v2 root had current/max/stat. spark-724a (7.0.0-1019-nvidia) exposes only
    memory.stat at the root and nothing memory.* in this session's scope, so walk the
    self-cgroup ancestors and, failing that, report no cgroup accounting instead of crashing
    or silently pretending to have a limit."""
    root=Path('/sys/fs/cgroup')
    if (root/'memory.current').is_file() and (root/'memory.max').is_file() and (root/'memory.stat').is_file():
        return root
    try:
        rel=next(l.split('::',1)[1].strip() for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
    except Exception:
        rel=''
    p=root/rel.lstrip('/')
    while True:
        if (p/'memory.current').is_file() and (p/'memory.max').is_file() and (p/'memory.stat').is_file():
            return p
        if p==root or p.parent==p: return None
        p=p.parent

def memory_sample():
    mi={p[0].rstrip(':'):int(p[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(p:=line.split())>=2}
    cg=cgroup_dir()
    if cg is None:
        # No cgroup memory limit for this process tree: the host is the budget, and it is
        # labelled so nobody reads a cgroup floor that was never enforced.
        return dict(available_gib=mi['MemAvailable']/GIB, free_gib=mi['MemFree']/GIB,
                    cached_gib=mi.get('Cached',0)/GIB, swap_used_gib=(mi['SwapTotal']-mi['SwapFree'])/GIB,
                    cgroup_current_gib=(mi['MemTotal']-mi['MemAvailable'])/GIB,
                    cgroup_headroom_gib=mi['MemAvailable']/GIB, cgroup_accounting=False,
                    psi_full10=next(float(w.split('=')[1]) for l in Path('/proc/pressure/memory').read_text().splitlines() if l.startswith('full ') for w in l.split() if w.startswith('avg10=')),
                    host_oom_kill=kv('/proc/vmstat')['oom_kill'], cgroup_oom_kill=0, cgroup_oom=0,
                    oom_kill=kv('/proc/vmstat')['oom_kill'], oom=0)
    s=kv(cg/'memory.stat'); ev=kv(cg/'memory.events')
    current=int((cg/'memory.current').read_text()); raw=(cg/'memory.max').read_text().strip()
    limit=int(raw) if raw!='max' else mi['MemTotal']
    # Inactive clean file cache is reclaimable; do not equate disk allocation with RAM use.
    reclaim=max(0,s.get('inactive_file',0)-s.get('file_dirty',0)-s.get('file_writeback',0))
    psi=Path('/proc/pressure/memory').read_text().splitlines()
    full=next(float(w.split('=')[1]) for l in psi if l.startswith('full ') for w in l.split() if w.startswith('avg10='))
    return dict(available_gib=mi['MemAvailable']/GIB, free_gib=mi['MemFree']/GIB,
                cached_gib=mi.get('Cached',0)/GIB, swap_used_gib=(mi['SwapTotal']-mi['SwapFree'])/GIB,
                cgroup_current_gib=current/GIB,cgroup_headroom_gib=(limit-current+reclaim)/GIB,cgroup_accounting=True,
                psi_full10=full,host_oom_kill=kv('/proc/vmstat')['oom_kill'],
                cgroup_oom_kill=ev['oom_kill'],cgroup_oom=ev['oom'],
                oom_kill=ev['oom_kill'],oom=ev['oom'])

def require_load_headroom(s, memory_policy='physical'):
    if memory_policy == 'uma':
        # Initial no-draft 2.50bpw load only: weights + staging + OS/cgroup reserve.
        if s['available_gib'] < 102 or s['cgroup_headroom_gib'] < 102 or s['free_gib'] < 2:
            raise RuntimeError('Insufficient UMA host/cgroup/emergency physical headroom')
        return
    if memory_policy != 'physical':
        raise ValueError('Unknown memory policy: '+str(memory_policy))
    # Conservative floor for this 98.5 GB pack before any CUDA/model initialization.
    if s['free_gib']<110:
        raise RuntimeError(f"Insufficient physical RAM for load: {s['free_gib']:.2f} GiB free; require 110 GiB")
    if s['available_gib']<110 or s['cgroup_headroom_gib']<110:
        raise RuntimeError('Insufficient host/cgroup headroom for load')

def parent_death():
    parent=os.getppid()
    if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0: os._exit(127)
    if os.getppid()!=parent: os._exit(127)

def supervise(command, outdir, sample=memory_sample, interval=.25, env=None, max_seconds=7200, memory_policy='physical'):
    if memory_policy not in ('physical', 'uma'):
        raise ValueError('Unknown memory policy: '+str(memory_policy))
    free_floor = 2 if memory_policy == 'uma' else 12
    cgroup_floor = 8 if memory_policy == 'uma' else 4
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    first=sample()
    physical_low = first['free_gib'] < free_floor and (memory_policy == 'physical' or min(first['available_gib'], first['cgroup_headroom_gib']) < 16)
    if first['available_gib']<8 or physical_low or first['cgroup_headroom_gib']<cgroup_floor:
        raise RuntimeError('Insufficient headroom before launch')
    started=time.monotonic(); child=None; reason='exit'; last=first
    minimum=first['available_gib']; minimum_free=first['free_gib']; requested=[]
    first_host=first.get('host_oom_kill')
    def request_stop(sig,frame): requested.append(sig)
    old={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with (outdir/'server.log').open('w') as log, (outdir/'memory.jsonl').open('w',buffering=1) as metrics:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                   start_new_session=True,preexec_fn=parent_death)
            (outdir/'pid.json').write_text(json.dumps({'guard_pid':os.getpid(),'server_pid':child.pid,'command':command}))
            while child.poll() is None:
                try:
                    last=sample(); minimum=min(minimum,last['available_gib']); minimum_free=min(minimum_free,last['free_gib'])
                    metrics.write(json.dumps({'elapsed_s':round(time.monotonic()-started,3),**last})+'\n')
                    # Clean-cache reclaim can leave low MemFree with abundant actual headroom.
                    physical_low = last['free_gib'] < free_floor and (memory_policy == 'physical' or min(last['available_gib'], last['cgroup_headroom_gib']) < 16)
                    if physical_low: reason='physical_headroom';break
                    if last['available_gib']<8: reason='host_headroom';break
                    if last['cgroup_headroom_gib']<cgroup_floor: reason='cgroup_headroom';break
                    if last['psi_full10']>10: reason='memory_pressure';break
                    if last['oom_kill']>first['oom_kill']: reason='oom_counter_changed';break
                    if first_host is not None and last.get('host_oom_kill') is not None and last['host_oom_kill']>first_host:
                        reason='host_oom_counter_changed';break
                    if requested or (outdir/'STOP').exists(): reason='requested_stop';break
                    if time.monotonic()-started>max_seconds: reason='time_limit';break
                    time.sleep(interval)
                except Exception as exc:
                    reason='monitor_error:'+repr(exc);break
    finally:
        if child and child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=.75)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait(timeout=10)
        for s,handler in old.items():signal.signal(s,handler)
    result=dict(reason=reason,returncode=child.returncode if child else None,elapsed_s=time.monotonic()-started,
                min_available_gib=minimum,min_free_gib=minimum_free,oom_kill_delta=last['oom_kill']-first['oom_kill'],
                host_oom_kill_delta=(last.get('host_oom_kill')-first_host) if (first_host is not None and last.get('host_oom_kill') is not None) else None)
    (outdir/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config');a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text()); env=os.environ.copy();env.update(cfg.get('env',{}))
    # spark-724a: the lock path is configurable; France hardcodes /workspace/mimo-tune/.
    lock_path = Path(cfg.get('lock_path', '/workspace/mimo-tune/server.lock'))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(supervise(cfg['command'],cfg['outdir'],env=env,max_seconds=cfg.get('max_seconds',7200),memory_policy=cfg.get('memory_policy','physical'))),flush=True)

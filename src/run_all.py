"""Run every fine-tuning job on shared GPUs.

The GPUs are shared with other users' jobs, so each job is placed on the GPU with the most
free memory, jobs that die (e.g. OOM when a neighbour grows) are retried, and finished jobs
(result json present) are skipped, which makes the script resumable.
"""
import os
import subprocess
import sys
import time

from langs import RESULTS, ROOT

SEEDS = {"ner": [1, 2, 3], "sib": [1, 2, 3, 4, 5], "xnli": [1, 2, 3]}
NEED_MB, PER_GPU, RESERVE_S, MAX_TRIES = 9000, 3, 120, 20

# cheap, many-language tasks first; XNLI (longest) last
jobs = [(t, m, s) for t in ("ner", "sib", "xnli") for s in SEEDS[t] for m in ("mbert", "xlmr")]
if len(sys.argv) > 1:
    jobs = [j for j in jobs if j[0] in sys.argv[1:]]
tag = lambda j: f"{j[0]}_{j[1]}_s{j[2]}"
done = lambda j: os.path.exists(os.path.join(RESULTS, "model_outputs", tag(j) + ".json"))


def free_mb():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    return [int(x) for x in out.decode().split()]


def alive(j):
    """True if a train.py process for this job exists (e.g. left over from an earlier scheduler)."""
    pat = f"{ROOT}/src/train.py --task {j[0]} --model {j[1]} --seed {j[2]}$"
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


# jobs still running under a previous scheduler are left alone and picked up once they exit
queue = [j for j in jobs if not done(j)]
tries, running = {}, []  # running: (job, gpu, popen, start_time)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
while queue or running:
    for r in list(running):
        job, gpu, p, t0 = r
        if p.poll() is None:
            continue
        running.remove(r)
        if done(job):
            print(f"OK   {tag(job)} gpu{gpu} {time.time()-t0:.0f}s", flush=True)
        else:
            tries[job] = tries.get(job, 0) + 1
            print(f"FAIL {tag(job)} gpu{gpu} try {tries[job]}", flush=True)
            if tries[job] < MAX_TRIES:
                queue.append(job)
    queue = [j for j in queue if not done(j)]
    if queue and alive(queue[0]):
        queue.append(queue.pop(0))
        if all(alive(j) for j in queue):
            time.sleep(10)
        continue
    if queue:
        free = free_mb()
        for g in range(len(free)):
            mine = [r for r in running if r[1] == g]
            # jobs started recently have not allocated their memory yet
            free[g] -= NEED_MB * sum(time.time() - r[3] < RESERVE_S for r in mine)
            if len(mine) >= PER_GPU:
                free[g] = 0
        g = max(range(len(free)), key=lambda i: free[i])
        if free[g] >= NEED_MB:
            job = queue.pop(0)
            log = open(os.path.join(ROOT, "logs", f"train_{tag(job)}.log"), "a")
            p = subprocess.Popen([sys.executable, os.path.join(ROOT, "src", "train.py"), "--task", job[0],
                                  "--model", job[1], "--seed", str(job[2])],
                                 env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=log, stderr=subprocess.STDOUT)
            running.append((job, g, p, time.time()))
            print(f"RUN  {tag(job)} gpu{g}", flush=True)
            continue
    time.sleep(10)
missing = [tag(j) for j in jobs if not done(j)]
print("ALL DONE" if not missing else f"MISSING {missing}", flush=True)

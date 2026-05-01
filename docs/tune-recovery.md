# Hung parallel tune: diagnose, kill, resume

When `tune` runs with `--parallel-trials` > 1, workers share one SQLite study at
`trading_system/models/meta_learner/ppo_tune_study.db`. If all workers go idle
and the DB stops updating, the run is wedged.

## 1. Confirm no progress

Check the study DB modification time (should advance as trials complete):

```bash
stat trading_system/models/meta_learner/ppo_tune_study.db
python -m trading_system tune status --study-name "YOUR_STUDY_NAME"
```

If `n_trials` and the DB `mtime` are frozen for longer than expected while
processes show 0% GPU and all threads sleeping, treat as hung.

## 2. Stack dumps with py-spy (root-owned processes need sudo)

Install (once):

```bash
.venv/bin/pip install py-spy
```

Find PIDs (parent is the `python -m trading_system tune` process; children show
`multiprocessing.spawn` / `spawn_main`):

```bash
ps aux | grep "trading_system tune"
```

Capture stacks (replace PIDs):

```bash
sudo mkdir -p /tmp/pyspy
PYSPY=/home/jpearce/Documents/StockBot3.0/.venv/bin/py-spy

for pid in PARENT_PID WORKER1 WORKER2 WORKER3 WORKER4; do
  echo "=== dumping $pid ==="
  sudo env "PATH=$PATH" "$PYSPY" dump --pid "$pid" --locals > "/tmp/pyspy/dump-$pid.txt" 2>&1
done
```

Review `/tmp/pyspy/dump-*.txt` for repeated frames on `queue`, `logging`,
`sqlite3`, `torch.cuda`, or `threading.Lock`.

## 3. Kill the wedged tree

Prefer SIGTERM on the parent so children exit cleanly:

```bash
sudo kill -TERM PARENT_PID
sleep 5
ps -p PARENT_PID,WORKER1,... -o pid,stat,cmd || true
sudo kill -KILL PARENT_PID WORKER1 ... 2>/dev/null
```

Completed Optuna trials are already persisted in the SQLite file.

## 4. Fix ownership if you ran tune as root

```bash
sudo chown "$USER:$USER" trading_system/models/meta_learner/ppo_tune_study.db*
sudo chown -R "$USER:$USER" trading_system/models/meta_learner/
```

## 5. Resume the same study (omit `--new`)

Use the same `--study-name` and flags as before. Example:

```bash
cd /path/to/StockBot3.0
nohup .venv/bin/python -u -m trading_system tune \
  --timesteps 2000000 \
  --tickers SPY,QQQ \
  --n-envs 32 --n-trials 200 --timeout 0 \
  --study-name "2M Historical 3" \
  --parallel-trials 4 --torch-vec \
  > trading_system/outputs/tune-resume.log 2>&1 &
echo "PID: $!"
```

## 6. Stall watchdog (optional)

For parallel tuning only, if `PPO_TUNE_STALL_TIMEOUT_SEC` is set to a positive
number of seconds, the parent process exits with code 2 when the study DB file
has had no `mtime` change for that long. Default is `0` (disabled).

Set high enough that a normal trial can finish without DB writes, or leave
disabled. Heartbeats on the Optuna storage still help recover dead RUNNING
trials; the watchdog is a last resort for total parent-visible stalls.

## 7. Code-side mitigations (already in tree)

- `get_tune_rdb_storage()` in `trading_system/meta_learner/tuner.py`: SQLite
  `connect_args.timeout=120`, plus Optuna `heartbeat_interval` / `grace_period`
  on `RDBStorage`.
- `enable_sqlite_wal()` in `trading_system/meta_learner/parallel_tuner.py`
  before parallel workers start.

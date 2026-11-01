#!/usr/bin/env python3
# Live, local-only dashboard for a Kangaroo4U DP table.
# Binds to the loopback addresses only, so nothing is exposed off this machine.
#   /           -> the dashboard page (index.html)
#   /data.json  -> headline numbers fresh per request, visuals from a cached scan
#
# Headline numbers come from the 256-byte table header, which the solver stamps
# on every save. Scanning the table body is a background job on a timer.
import http.server, socketserver, socket, struct, math, json, os, re, threading, time, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "index.html")

NJUMPS   = 512               # JMP_CNT in the solver
SCAN_SEC = 300               # rescan no more often than the solver saves
SAMPLE_TARGET = 250000       # cap per-scan python work
GRID_C, GRID_R = 72, 40      # collision-window lens, in cells
LOCAL_FRESH = 120            # a live solver prints a speed line every 10 s
HIST_SEC  = 60               # one rate/count sample a minute
HIST_KEEP = 2880             # ... kept for 48 h

# Filled in by configure() from settings.ini.
P = {}


def configure(paths, name, rng, dp, port):
    P.update(paths)
    P.update(name=name, range=rng, dp=dp, port=port)


# ---------- process liveness (Linux and Windows) ----------
def _pid_alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259     # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def solver_pid():
    try:
        pid = int(open(P["pidfile"]).read().strip())
    except Exception:
        return None
    return pid if _pid_alive(pid) else None


# ---------- header (cheap, every request) ----------
def _need(rng, dp, count, ops):
    need = 1.15 * math.sqrt(2.0 ** rng)
    return dict(ops_needed=need, ops_progress=(ops / need) if need else 0.0,
                dps_needed=need / (1 << dp), progress=count / (need / (1 << dp)))

def read_header():
    path = P["table"]
    if not os.path.exists(path):
        return dict(range_bits=P["range"], dpBits=P["dp"], nJumps=NJUMPS, count=0,
                    banked_ops=0, saved_at=0, store=os.path.basename(path), exists=False,
                    **_need(P["range"], P["dp"], 0, 0))
    with open(path, "rb") as f:
        h = f.read(256)
    if len(h) < 256 or h[2] != 0xDB:
        raise ValueError("not a DP table: " + os.path.basename(path))
    rng, dp = h[0], h[1]
    count, ops, saved = struct.unpack_from("<QQQ", h, 72)
    exact = (os.path.getsize(path) - 256 - 2 * (256 ** 3)) // 32
    if exact >= 0 and (count == 0 or count != exact):
        count = exact
    return dict(range_bits=rng, dpBits=dp, nJumps=NJUMPS, count=count,
                banked_ops=ops, saved_at=saved, store=os.path.basename(path), exists=True,
                **_need(rng, dp, count, ops))


# ---------- GPUs (nvidia-smi) ----------
_gpu = {"t": 0.0, "d": None}

def _num(x):
    try: return float(x)
    except Exception: return None

def read_gpus():
    now = time.time()
    if _gpu["d"] is not None and now - _gpu["t"] < 3:
        return _gpu["d"]
    try:
        q = ("index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,"
             "utilization.gpu,fan.speed,clocks.sm")
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        cards = []
        for line in out.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 10:
                continue
            cards.append(dict(index=f[0], name=f[1], temp=_num(f[2]), power=_num(f[3]),
                              power_limit=_num(f[4]), mem_used=_num(f[5]), mem_total=_num(f[6]),
                              util=_num(f[7]), fan=_num(f[8]), clock=_num(f[9])))
        d = {"cards": cards} if cards else {"error": "nvidia-smi returned no GPUs"}
    except Exception as e:
        d = {"error": str(e)}
    _gpu.update(t=now, d=d)
    return d


# ---------- solver ----------
# A speed line outlives the run that wrote it, so the log alone cannot say
# whether the solver is working. The rate counts only when the solver process is
# alive AND its log is still moving.
def _log_tail(n=8000):
    with open(P["log"], "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - n))
        return f.read().decode("latin-1")

def read_local():
    pid = solver_pid()
    d = {"status": "idle", "mkeys": None, "last_mkeys": None, "last_at": None,
         "age": None, "err": None, "pid": pid}
    try:
        st = os.stat(P["log"])
        data = _log_tail()
    except Exception:
        return d
    d["age"] = max(0.0, time.time() - st.st_mtime)
    d["last_at"] = int(st.st_mtime)
    for seg in re.split(r"[\r\n]", data):
        m = re.search(r"Speed:\s*(\d+)\s*MKeys/s(?:,\s*Err:\s*(\d+))?", seg)
        if m:
            d["last_mkeys"] = float(m.group(1))
            if m.group(2): d["err"] = int(m.group(2))
    if pid:
        if d["age"] > LOCAL_FRESH:
            d["status"] = "stalled"
        elif d["last_mkeys"] is None:
            d["status"] = "starting"
        else:
            d["status"] = "producing"
            d["mkeys"] = d["last_mkeys"]
    return d

def read_rate():
    loc = read_local()
    return {"rate_mjs": loc["mkeys"], "local": loc}

def read_solved():
    found = False
    try:
        found = bool(re.search(r"PRIVATE KEY:\s*[0-9A-Fa-f]+",
                               open(P["results"], encoding="latin-1").read()))
    except Exception:
        pass
    if not found:
        try:
            found = bool(re.search(r"PRIVATE KEY:\s*[0-9A-Fa-f]+", _log_tail()))
        except Exception:
            pass
    if not found:
        return {"solved": False}
    try:
        at = int(os.path.getmtime(P["results"]))
    except Exception:
        at = None
    return {"solved": True, "solved_at": at, "results": P["results"]}


# ---------- full scan (background, cached) ----------
# Layout: 256B header, then 256^3 lists of {u16 cnt, cnt * 32B records}.
# A record is DBRec minus its first 3 bytes, which are the list indices:
#   x[3:12] (9B) | d (22B) | type (1B)
# x is stored little-endian, so the three list-index bytes are the LOWEST bytes
# of x. Two charts depend on that:
#   xhist  byte 11 of x, which nothing indexes on, so it is the honest test of
#          whether the walk is spreading out
#   ihist  byte 0 of x, the outer list index, counted exactly from the list
#          headers: table occupancy rather than walk quality.
#
# Stored distances are SIGNED and are not interval positions. Kangaroos are
# seeded inside [0, 34) units of 2^(range-5) but keep accumulating distance as
# they walk, so the window follows the data rather than staying fixed.
BAND_LO, BAND_HI = -128.0, 192.0
_band = [BAND_LO, BAND_HI]
_snap = {"cells": [], "both_cells": 0, "tame": 0, "wild": 0,
         "cov_tame": [0] * 240, "cov_wild": [0] * 240, "xhist": [0] * 256,
         "ihist": [0] * 256,
         "dmin_u": 0.0, "dmax_u": 0.0, "cov_lo": BAND_LO, "cov_hi": BAND_HI,
         "tame_out": 0, "wild_out": 0, "sampled": False, "step": 1,
         "gridN_c": GRID_C, "gridN_r": GRID_R,
         "scan_at": 0, "scanning": False}
_lock = threading.Lock()

WIDE_LO, WIDE_HI = -1024, 1024      # the auxiliary histogram, 1 unit per bin
BAND_Q = 0.002                      # trim this much off each tail

def retune(wide, below, above):
    tot = sum(wide) + below + above
    if tot < 100:
        return
    want_lo, want_hi = tot * BAND_Q, tot * (1.0 - BAND_Q)
    run = below
    qlo = qhi = None
    for i, v in enumerate(wide):
        if qlo is None and run + v >= want_lo:
            qlo = WIDE_LO + i
        run += v
        if run >= want_hi:
            qhi = WIDE_LO + i + 1
            break
    if qlo is None: qlo = WIDE_LO
    if qhi is None: qhi = WIDE_HI
    span = max(qhi - qlo, 16.0)
    _band[0] = math.floor((qlo - 0.08 * span) / 8.0) * 8.0
    _band[1] = math.ceil((qhi + 0.08 * span) / 8.0) * 8.0

def scan():
    gc, gr, BINS = GRID_C, GRID_R, 240
    COV_LO, COV_HI = _band
    grid = {}; xh = [0] * 256; ih = [0] * 256; tb = [0] * BINS; wb = [0] * BINS
    wide = [0] * (WIDE_HI - WIDE_LO); w_below = w_above = 0
    tame = wild = 0; dmin = None; dmax = 0; seen = 0; tout = wout = 0
    hdr = read_header()
    if not hdr["exists"]:
        return
    total = hdr["count"]
    unit = float(1 << (hdr["range_bits"] - 5))
    step = max(1, total // SAMPLE_TARGET)

    def emit(i, j, k, x3, x11, dv, t):
        nonlocal tame, wild, seen, dmin, dmax, w_below, w_above, tout, wout
        if t: wild += 1
        else: tame += 1
        seen += 1
        if seen % step: return
        d = dv
        if d >> 175: d -= 1 << 176      # two's complement
        u = d / unit
        if dmin is None or u < dmin: dmin = u
        if u > dmax: dmax = u
        wi = int(math.floor(u)) - WIDE_LO
        if wi < 0: w_below += 1
        elif wi >= len(wide): w_above += 1
        else: wide[wi] += 1
        b = int((u - COV_LO) / (COV_HI - COV_LO) * BINS)
        if 0 <= b < BINS:
            (tb if t == 0 else wb)[b] += 1
        else:
            if t: wout += 1
            else: tout += 1
        c = (j * 256 + k) % gc
        r = (i * 256 + x3) % gr
        cell = grid.get((c, r))
        if cell is None: cell = [0, 0]; grid[(c, r)] = cell
        cell[0 if t == 0 else 1] += 1
        xh[x11] += 1

    with open(P["table"], "rb") as f:
        f.seek(256)
        buf = b""; pos = 0
        def need(n):
            nonlocal buf, pos
            if len(buf) - pos < n:
                buf = buf[pos:] + f.read(max(1 << 22, n))
                pos = 0
            return len(buf) - pos >= n
        for i in range(256):
            for j in range(256):
                for k in range(256):
                    if not need(2): return
                    cnt = buf[pos] | (buf[pos + 1] << 8); pos += 2
                    if not cnt: continue
                    ih[i] += cnt
                    if not need(cnt * 32): return
                    for m in range(cnt):
                        p = pos + m * 32
                        emit(i, j, k, buf[p], buf[p + 8],
                             int.from_bytes(buf[p + 9:p + 31], "little"),
                             buf[p + 31])
                    pos += cnt * 32
    cells = [[c, r, v[0], v[1]] for (c, r), v in grid.items()]
    retune(wide, w_below, w_above)
    with _lock:
        _snap.update(cells=cells, both_cells=sum(1 for c in cells if c[2] and c[3]),
                     tame=tame, wild=wild, cov_tame=tb, cov_wild=wb, xhist=xh, ihist=ih,
                     dmin_u=(dmin or 0), dmax_u=dmax, cov_lo=COV_LO, cov_hi=COV_HI,
                     band_next=list(_band),
                     tame_out=tout, wild_out=wout, gridN_c=gc, gridN_r=gr,
                     sampled=(step > 1), step=step, scan_at=int(time.time()))

def scanner():
    last = None
    while True:
        try:
            st = os.stat(P["table"]) if os.path.exists(P["table"]) else None
            key = (st.st_mtime, st.st_size) if st else None
            if key and key != last:
                with _lock: _snap["scanning"] = True
                before = tuple(_band)
                scan()
                if tuple(_band) != before:
                    scan()      # the window moved, so redraw with it
                last = key
        except Exception:
            pass
        finally:
            with _lock: _snap["scanning"] = False
        # the first table lands a few minutes into a fresh run, so look again
        # soon until there is something to scan
        time.sleep(SCAN_SEC if last else 30)


# ---------- rate/count history (background, persisted) ----------
_hist = []
_hlock = threading.Lock()

def hist_load():
    rows = []
    try:
        for line in open(P["history"]):
            p = line.strip().split(",")
            if len(p) == 3:
                rows.append((int(p[0]), float(p[1]), int(p[2])))
    except Exception:
        pass
    with _hlock:
        _hist[:] = rows[-HIST_KEEP:]

def hist_sampler():
    hist_load()
    while True:
        try:
            r = read_rate()
            row = (int(time.time()), r["rate_mjs"] or 0.0, read_header()["count"])
            with _hlock:
                _hist.append(row)
                del _hist[:-HIST_KEEP]
                rows = list(_hist)
            tmp = P["history"] + ".tmp"
            with open(tmp, "w") as f:
                f.writelines("%d,%.1f,%d\n" % r for r in rows)
            os.replace(tmp, P["history"])
        except Exception:
            pass
        time.sleep(HIST_SEC)

def hist_series(n=240):
    with _hlock:
        h = list(_hist)
    if len(h) > n:                       # keep the newest sample, thin the rest
        step = -(-len(h) // n)
        h = list(reversed(h[::-1][::step]))
    return {"t": [r[0] for r in h], "rate": [r[1] for r in h],
            "count": [r[2] for r in h], "span": HIST_KEEP * HIST_SEC}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        try:
            if self.path.startswith("/data"):
                d = read_header()
                with _lock: d.update(_snap)
                d.update(read_rate())
                d["history"] = hist_series()
                d.update(read_solved())
                d["gpu"] = read_gpus()
                d["name"] = P["name"]
                d["port"] = P["port"]
                self._send(json.dumps(d).encode(), "application/json")
            else:
                self._send(open(HTML, "rb").read(), "text/html; charset=utf-8")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            self._send(json.dumps({"error": str(e)}).encode(), "application/json")
    def log_message(self, *a):
        pass


class _V6(socketserver.ThreadingTCPServer):
    address_family = socket.AF_INET6
    daemon_threads = True


class _V4(socketserver.ThreadingTCPServer):
    daemon_threads = True


def start():
    """Serve the dashboard on background threads. Returns False if no port could be bound."""
    # Listen on both loopback addresses: a browser that resolves "localhost" to
    # ::1 first is refused by an IPv4-only listener. Windows lets a second
    # process bind a port already in use when SO_REUSEADDR is set, so only ask
    # for it elsewhere.
    socketserver.TCPServer.allow_reuse_address = (os.name != "nt")
    servers = []
    for cls, addr in ((_V6, ("::1", P["port"])), (_V4, ("127.0.0.1", P["port"]))):
        try:
            servers.append(cls(addr, Handler))
        except OSError:
            pass
    if not servers:
        return False
    threading.Thread(target=scanner, daemon=True).start()
    threading.Thread(target=hist_sampler, daemon=True).start()
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    return True


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    import kangaroo4u
    kangaroo4u.main(["dashboard"] + sys.argv[1:])

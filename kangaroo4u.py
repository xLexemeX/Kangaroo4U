#!/usr/bin/env python3
"""Kangaroo4U launcher: runs the GPU solver and the live dashboard together.

    run.sh / run.bat               start the search and the dashboard
    run.sh dashboard               the dashboard only
    run.sh bench                   solve random 76-bit keys to check the GPU works

Press Ctrl+C once to stop. The solver saves its table before it exits.
"""
import configparser, os, re, signal, subprocess, sys, threading, time, webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
EXE = os.path.join(ROOT, "bin", "rckangaroo.exe" if os.name == "nt" else "rckangaroo")
LOG_ROTATE_BYTES = 50 * 1024 * 1024

sys.path.insert(0, os.path.join(ROOT, "dashboard"))
import server as dashboard  # noqa: E402


def say(msg):
    print("[kangaroo4u] " + msg, flush=True)


def load_settings():
    path = os.path.join(ROOT, "settings.ini")
    cp = configparser.ConfigParser()
    if not cp.read(path, encoding="utf-8"):
        raise SystemExit("settings.ini not found next to kangaroo4u.py")
    s = cp["search"]
    cfg = dict(
        name=s.get("name", "").strip() or "Kangaroo search",
        pubkey=s.get("pubkey").strip(),
        start=s.get("start").strip(),
        range=s.getint("range"),
        dp=s.getint("dp"),
        gpus=cp.get("gpu", "gpus", fallback="").strip(),
        power=cp.get("gpu", "power_limit_watts", fallback="").strip(),
        memclk=cp.get("gpu", "memory_clock_mhz", fallback="").strip(),
        savesec=cp.getint("solver", "save_every_seconds", fallback=300),
        port=cp.getint("dashboard", "port", fallback=8899),
        browser=cp.getboolean("dashboard", "open_browser", fallback=True),
    )
    if not re.fullmatch(r"0[23][0-9a-fA-F]{64}|04[0-9a-fA-F]{128}", cfg["pubkey"]):
        raise SystemExit("settings.ini: pubkey must be a compressed or uncompressed public key in hex")
    if not re.fullmatch(r"[0-9a-fA-F]+", cfg["start"]):
        raise SystemExit("settings.ini: start must be hex")
    if not 32 <= cfg["range"] <= 170:
        raise SystemExit("settings.ini: range must be 32..170")
    if not 14 <= cfg["dp"] <= 32:
        raise SystemExit("settings.ini: dp must be 14..32")
    if cfg["gpus"] and not cfg["gpus"].isdigit():
        raise SystemExit("settings.ini: gpus is a list of digits, e.g. 01 for GPUs 0 and 1")
    return cfg


def paths(cfg):
    # The table is named after the key and parameters it belongs to, so changing
    # settings.ini starts a new table instead of being refused by the old one.
    tag = "%s-r%d-dp%d" % (cfg["pubkey"][:12].lower(), cfg["range"], cfg["dp"])
    return dict(
        table=os.path.join(DATA, "table-" + tag + ".rcdb"),
        log=os.path.join(DATA, "solver.log"),
        pidfile=os.path.join(DATA, "solver.pid"),
        history=os.path.join(DATA, "history-" + tag + ".csv"),
        results=os.path.join(DATA, "RESULTS.TXT"),
    )


def start_dashboard(cfg, p, open_browser):
    dashboard.configure(p, cfg["name"], cfg["range"], cfg["dp"], cfg["port"])
    url = "http://localhost:%d" % cfg["port"]
    if dashboard.start():
        say("dashboard: " + url)
        if open_browser:
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        return True
    say("dashboard: port %d is busy (already open in another window?), "
        "change [dashboard] port in settings.ini to run a second one" % cfg["port"])
    return False


def nvsmi(args):
    cmd = ["nvidia-smi"] + args
    if os.name != "nt":
        cmd = ["sudo", "-n"] + cmd
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def tune_gpus(cfg):
    sel = ["-i", ",".join(cfg["gpus"])] if cfg["gpus"] else []
    need = "an Administrator window" if os.name == "nt" else "passwordless sudo for nvidia-smi"
    if cfg["power"]:
        if nvsmi(sel + ["-pl", cfg["power"]]):
            say("power limit set to %s W" % cfg["power"])
        else:
            say("could not set the power limit (needs %s), running at the card's own limit" % need)
    if cfg["memclk"]:
        if nvsmi(sel + ["-lmc", cfg["memclk"]]):
            say("memory clock locked to %s MHz" % cfg["memclk"])
        else:
            say("could not lock the memory clock (needs %s), running at stock" % need)


def running_pid(p):
    dashboard.P.update(p)
    return dashboard.solver_pid()


def rotate_log(log):
    try:
        if os.path.getsize(log) > LOG_ROTATE_BYTES:
            os.replace(log, log + ".old")
    except OSError:
        pass


def run_solver(args, p, log_path, cwd):
    """Run the solver in the foreground, echoing its output to the console and the log."""
    stop = {"asked": 0}

    def on_stop(sig, frame):
        # The solver shares this console, so Ctrl+C reaches it directly; this
        # only keeps the launcher alive until it has finished saving.
        stop["asked"] += 1
        if stop["asked"] == 1:
            say("stopping, saving the table (this can take a minute)...")
        elif os.name != "nt":
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass

    def on_term(sig, frame):
        on_stop(sig, frame)
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass

    rotate_log(log_path)
    log = open(log_path, "ab", buffering=0)
    log.write(("\r\n==== %s  %s\r\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                        " ".join(os.path.basename(a) for a in args))).encode())
    proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(p["pidfile"], "w") as f:
        f.write(str(proc.pid))
    signal.signal(signal.SIGINT, on_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_stop)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGHUP, on_term)

    tail = b""
    out = sys.stdout.buffer
    try:
        while True:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                break
            log.write(chunk)
            try:
                out.write(chunk); out.flush()
            except Exception:
                pass
            tail = (tail + chunk)[-4096:]
        proc.wait()
    finally:
        log.close()
        try:
            os.remove(p["pidfile"])
        except OSError:
            pass
    return proc.returncode, tail.decode("latin-1")


def cmd_run(cfg, p):
    if not os.path.exists(EXE):
        raise SystemExit("solver not built: %s is missing. Run build.%s first."
                         % (EXE, "bat" if os.name == "nt" else "sh"))
    pid = running_pid(p)
    if pid:
        say("the solver is already running (pid %d). Open http://localhost:%d to watch it."
            % (pid, cfg["port"]))
        return 0

    say("search  %s" % cfg["name"])
    say("pubkey  %s" % cfg["pubkey"])
    say("range   %d bits from 0x%s, dp %d" % (cfg["range"], cfg["start"], cfg["dp"]))
    say("table   %s" % p["table"])
    tune_gpus(cfg)
    start_dashboard(cfg, p, cfg["browser"])

    args = [EXE, "-pubkey", cfg["pubkey"], "-start", cfg["start"],
            "-range", str(cfg["range"]), "-dp", str(cfg["dp"]),
            "-db", p["table"], "-savesec", str(cfg["savesec"])]
    if cfg["gpus"]:
        args += ["-gpu", cfg["gpus"]]
    code, tail = run_solver(args, p, p["log"], DATA)

    m = re.search(r"PRIVATE KEY:\s*([0-9A-Fa-f]+)", tail)
    if m:
        bar = "*" * 78
        print("\n" + bar + "\n  KEY FOUND: " + m.group(1) + "\n  saved to " + p["results"]
              + "\n" + bar + "\n", flush=True)
    else:
        say("solver stopped (exit code %s). The table is kept; run again to continue." % code)
    return 0


def cmd_bench(cfg, p):
    if not os.path.exists(EXE):
        raise SystemExit("solver not built: %s is missing" % EXE)
    say("benchmark: solving random 76-bit keys. Each solved key proves the GPU path works. Ctrl+C to stop.")
    args = [EXE, "-dp", "16", "-range", "76"]
    if cfg["gpus"]:
        args += ["-gpu", cfg["gpus"]]
    run_solver(args, dict(p, pidfile=os.path.join(DATA, "bench.pid")),
               os.path.join(DATA, "bench.log"), DATA)
    return 0


def cmd_dashboard(cfg, p):
    if not start_dashboard(cfg, p, cfg["browser"]):
        return 1
    say("Ctrl+C to close the dashboard (the solver, if running, is not affected)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    return 0


def main(argv):
    cmd = argv[0] if argv else "run"
    os.makedirs(DATA, exist_ok=True)
    cfg = load_settings()
    p = paths(cfg)
    if cmd in ("run", "start"):
        return cmd_run(cfg, p)
    if cmd == "dashboard":
        return cmd_dashboard(cfg, p)
    if cmd == "bench":
        return cmd_bench(cfg, p)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

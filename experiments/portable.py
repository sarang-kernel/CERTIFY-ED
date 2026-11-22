"""Platform layer of the experiments: memory cap, peak resident memory, CPU model and RAM.

POSIX keeps the behaviour the experiments had before this module existed (RLIMIT_AS through `resource`,
`ru_maxrss`, /proc), except that the peak resident memory on Linux is VmHWM of /proc/self/status (peak_rss_mb). Windows has no `resource` module and no preexec_fn; there the cap is a Job Object
with a per-process memory limit (JOB_OBJECT_LIMIT_PROCESS_MEMORY: committed private memory, the closest
analogue of RLIMIT_AS; an allocation beyond it fails and numpy raises MemoryError), peak memory is the
peak working set (GetProcessMemoryInfo), the CPU model comes from the registry and RAM from
GlobalMemoryStatusEx. Every function reports what it did instead of failing silently: cap_memory returns
the mechanism used, or a reason string starting with "not enforced".
"""
import os
import sys

WINDOWS = sys.platform == "win32"
_JOBS = []          # keep job handles alive for the life of the process


def posix_preexec(nbytes):
    """preexec_fn that caps a child's address space at nbytes (POSIX), or None on Windows."""
    if WINDOWS:
        return None

    def _limit():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
    return _limit


def cap_memory(nbytes):
    """Cap the memory of the calling process at nbytes. Returns the mechanism ('RLIMIT_AS',
    'job object') or 'not enforced: <reason>'."""
    if not WINDOWS:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        return "RLIMIT_AS"
    try:
        return _windows_job_cap(nbytes)
    except Exception as e:                       # e.g. a parent job that forbids nesting
        return f"not enforced: {type(e).__name__}: {e}"


def _windows_job_cap(nbytes):
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x00000100        # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(nbytes)
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # Extended
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
    _JOBS.append(job)
    return "job object"


def peak_rss_mb():
    """Peak resident memory of the calling process in MiB. Linux: VmHWM of /proc/self/status, the peak of the process's
    own address space. ru_maxrss is not read there: after fork (or vfork) and exec it starts at the peak resident size
    of the parent, so a child of a large process (a test inside a long pytest run) reported the parent's peak and missed
    its own below it. ru_maxrss where /proc is absent (KiB; bytes on macOS)."""
    if not WINDOWS:
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmHWM:"):
                            return int(line.split()[1]) / 1024
            except (OSError, ValueError, IndexError):
                pass
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 20 if sys.platform == "darwin" else r / 1024     # macOS reports bytes
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return float("nan")
    return c.PeakWorkingSetSize / 2 ** 20


def cpu_model():
    """CPU model string, or None."""
    try:
        if WINDOWS:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                return next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), None)
        import platform
        return platform.processor() or None
    except Exception:
        return None


def memory_gib():
    """Installed RAM in GiB (one decimal), or None."""
    try:
        if WINDOWS:
            import ctypes
            from ctypes import wintypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(m)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            return round(m.ullTotalPhys / 2 ** 30, 1)
        with open("/proc/meminfo") as f:
            kb = next(int(line.split()[1]) for line in f if line.startswith("MemTotal"))
        return round(kb / 2 ** 20, 1)
    except Exception:
        return None

"""lint_docs.py: house-style and safety checks for the documents of the project record.

Usage:  python tools/lint_docs.py FILE.md [FILE.md ...]        (or a directory: every .md below it)
        python tools/lint_docs.py --bundle SOURCES.txt FILE.md  (also report numbers of 3 or more significant
                                                                 characters that do not occur in the bundle)

The rules are those of docs/project/STYLE.md, adapted from the style checker of the mm3x3 release (lint_paper.py):
no list lines, no em or en dashes and no "--" used as a dash, no promotional or filler vocabulary, no
mention of the writing process (assistants, agents, sessions), no claim that a statement is "verified" or
"proven" (a statement is kernel-checked, machine-checked, or checked by a named person; D-12), and never the
private e-mail address of the author. Fenced code blocks and inline code spans are skipped, so file names and
record fields such as `spin1_landscape` do not trip the word list. Markdown pipe tables are allowed.
Exit code 0 means clean.
"""
import os
import re
import sys

BANNED = """leverage utilize utilise robust comprehensive delve seamless cutting-edge state-of-the-art groundbreaking
game-changing harness unlock empower synergy holistic paradigm tapestry realm intricate meticulous showcase underscore
foster navigate journey embark elevate streamline transformative innovative revolutionize revolutionise revolutionary
vibrant profound notably furthermore moreover crucial crucially pivotal powerful exciting remarkable boasts bespoke
multifaceted nuanced myriad plethora essentially arguably unveil spearhead landscape rigorous rigorously""".split()
PHRASES = ["deep dive", "key insight", "it is important to note", "it is worth noting", "plays a key role",
           "a testament to", "in conclusion", "proven correct"]
PROCESS = ["claude", "chatgpt", "llm", "language model", "sub-agent", "subagent", "agent", "assistant", "chat session",
           "session", "prompt", "artifact store", "tool call"]
CLAIM = ["verified", "proven"]
# The private address of the author is never written here: a candidate "local part@" is compared by its SHA-256.
_PRIVATE_SHA256 = "c8491228ab3de827c6ef569bf05ef7a5222512e0ce0a2fabef89719783e58b16"
_LOCAL_PART = re.compile(r"[a-z0-9._%+-]+@", re.I)


class _PrivateAddress:
    """search(text) returns a match when text contains the private address of the author, else None."""

    def search(self, text):
        import hashlib
        for m in _LOCAL_PART.finditer(text):
            s = m.group(0).lower()
            if any(hashlib.sha256(s[k:].encode()).hexdigest() == _PRIVATE_SHA256 for k in range(len(s) - 1)):
                return m
        return None


PRIVATE = _PrivateAddress()
LIST = re.compile(r"^([-*+\u2022\u25cf]\s|\d+[.)]\s|\([a-z0-9]+\)\s)")
DASHES = ("\u2014", "\u2013", "\u2012", "\u2015")


def strip_inline_code(line):
    return re.sub(r"`[^`]*`", "``", line)


def lint(text, bundle=None, allow_process=False):
    probs = []
    in_code = False
    for i, raw in enumerate(text.splitlines(), 1):
        if PRIVATE.search(raw):
            probs.append((i, "private address", raw[:80]))
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = strip_inline_code(raw)
        s = line.lstrip()
        if LIST.match(s):
            probs.append((i, "list line", raw[:80]))
        if any(d in line for d in DASHES):
            probs.append((i, "em/en dash", raw[:80]))
        if re.search(r"\s--\s|\w--\w", line) and not s.startswith("|"):
            probs.append((i, "double-hyphen dash", raw[:80]))
        low = re.sub(r"\]\([^)]*\)", "]()", line.lower())       # link targets are paths, not prose
        for w in BANNED:
            if re.search(r"\b" + re.escape(w) + r"\w*\b", low):
                probs.append((i, "banned word: " + w, raw[:80]))
        for p in PHRASES:
            if p in low:
                probs.append((i, "banned phrase: " + p, raw[:80]))
        for w in CLAIM:
            if re.search(r"\b" + w + r"\b", low):
                probs.append((i, "claim word: " + w, raw[:80]))
        if not allow_process:
            for w in PROCESS:
                if re.search(r"\b" + re.escape(w) + r"s?\b", low):
                    probs.append((i, "process word: " + w, raw[:80]))
    if bundle is not None:
        b = bundle.replace(",", "")
        for i, line in enumerate(text.splitlines(), 1):
            for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", strip_inline_code(line)):
                m = n.replace(",", "").rstrip(".")
                if len(m.replace(".", "")) >= 3 and m not in b:
                    probs.append((i, "number not in sources: " + n, line[:80]))
    return probs


def files(args):
    for a in args:
        if os.path.isdir(a):
            for dp, dns, fns in os.walk(a):
                dns[:] = [d for d in dns if d not in (".git", ".lake", "archive_v5", "__pycache__")]
                for f in sorted(fns):
                    if f.endswith(".md"):
                        yield os.path.join(dp, f)
        else:
            yield a


if __name__ == "__main__":
    args = sys.argv[1:]
    bundle = None
    if "--bundle" in args:
        k = args.index("--bundle")
        bundle = open(args[k + 1], encoding="utf-8").read()
        del args[k:k + 2]
    total = 0
    for f in files(args):
        text = open(f, encoding="utf-8-sig").read()
        allow = os.path.basename(f) in ("README.md", "STYLE.md", "AGENT_MATERIAL.md")
        P = lint(text, bundle, allow_process=allow)
        for p in P:
            print(f, *p, sep="\t")
        total += len(P)
    print(f"{total} problems")
    sys.exit(1 if total else 0)

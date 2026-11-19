"""lint_docs.py: house-style and safety checks for the documents of the project record.

Usage:  python tools/lint_docs.py FILE.md [FILE.md ...]        (or a directory: every .md below it)
        python tools/lint_docs.py --bundle SOURCES.txt FILE.md  (also report numbers of 3 or more significant
                                                                 characters that do not occur in the bundle)

The rules are those of docs/project/STYLE.md, adapted from the style checker of the mm3x3 release (lint_paper.py):
no list lines, no em or en dashes and no "--" used as a dash, no promotional or filler vocabulary, no
mention of the writing process (tools, sessions, prompts), no claim that a statement is "verified" or
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
PROCESS = ["sub-agent", "subagent", "agent", "assistant", "chat session", "session", "prompt", "artifact store",
           "tool call"]
# Further process words (product and tool names), compared by their SHA-256 digests in lower case so that this file
# does not spell them; a two-word entry is digested as the two words joined by one space.
_PROCESS_SHA256 = frozenset((
    "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d",
    "c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131",
    "60965168ce762e949600281ba6d01fee136e5b6e8257b1f216f9025ed324474c",
    "053ea4804ef1bb33d4a3d6fb024a614b6d257cebc2bc7cd915da9c9522f37ffc",
    "7d3194f79e645c42e4396dda38be04766810ec6a00d00aced3ffc2a0a1f1a9ef",
    "5d72436256ada53828b51895a94bb8489e9f1ac4fe937a8024ef1594e7045ff6",
    "3ea125d0bff386e6754b3782b300016fc79a9cf8f8669c0a5c3db64467ddb681",
    "487b91042c7cf27a19e23ea8699f5f354b1a0c3af9e418138dc6150d830f970d",
    "5eff592cc497fccb48ebfffaca38a83f9167bc1661b4477e1fc1b66b658cd5a0",
    "e12ce8285efc67c6d93d3a122e2589ed95089bcbb775ba5634d94e2b8385db07",
    "c9ad8f2cc1294afa0ef22fc2c019ff7243cdd272b2147ddca3f34c5036b05768",
    "5a5110ebe1544b31e6d4207a419bc1deb148735f89ae984ff0a2ab809b51f1df",
    "09cf980b5ff304ac11b7f6d2c5c263da2a867425798ef5cc5d2ebcf55c4fcd23",
))
# Upper-case abbreviations compared by digest with their case (a code identifier in lower case is not one of them).
_CASED_SHA256 = frozenset(("11fb682be0a0233d5fb899721ecfc1827d20f0d2ff2e093310efa61efca8af1c",))
_WORD = re.compile(r"[a-z0-9]+")
_UPPER = re.compile(r"\b[A-Z][A-Z0-9]+\b")
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


def _process_digest_hit(low, line=""):
    """True when a word of `low`, or a pair of adjacent words, has its digest in _PROCESS_SHA256 (a final s dropped),
    or an upper-case word of `line` has its digest in _CASED_SHA256."""
    import hashlib
    if any(hashlib.sha256(w.encode()).hexdigest() in _CASED_SHA256 for w in _UPPER.findall(line)):
        return True
    words = _WORD.findall(low)
    cands = set()
    for k, w in enumerate(words):
        cands.update((w, w[:-1] if w.endswith("s") else w))
        if k + 1 < len(words):
            nxt = words[k + 1]
            cands.update((w + " " + nxt, w + " " + (nxt[:-1] if nxt.endswith("s") else nxt)))
    return any(hashlib.sha256(c.encode()).hexdigest() in _PROCESS_SHA256 for c in cands)


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
            if _process_digest_hit(low, line):
                probs.append((i, "process word (listed by digest)", raw[:80]))
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

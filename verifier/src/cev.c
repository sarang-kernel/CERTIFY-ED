/* cev.c - independent verifier for CERTIFY-ED certificates (certificate/2.1 + .arrays.npz sidecar).
 *
 * Second-language re-check of the arithmetic a certificate records: hashes (body, arrays, spec, operator),
 * the S2 enclosure arithmetic (Theorem-1 radii from stored vectors and the producer operator when both are
 * available, the recorded cluster arithmetic otherwise), the S3 count-and-ordering logic from the stored
 * enclosures and the recorded inertia counts, and the consistency of the recorded statuses with what those
 * recomputations permit. No code is shared with certify_ed; the only inputs are the certificate files.
 *
 * Directed rounding: every bound is evaluated with fesetround(FE_UPWARD) / fesetround(FE_DOWNWARD) on IEEE-754
 * binary64 (SSE2); compile with -frounding-math -ffp-contract=off (verifier/build.py does).
 *
 * Exit codes: 0 every check passed (SKIP allowed), 1 at least one FAIL, 2 the inputs could not be read.
 * See verifier/README.md for the list of checks and their ids.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <fenv.h>
#include <errno.h>

#pragma STDC FENV_ACCESS ON

#define CEV_VERSION "0.1.0"

/* ------------------------------------------------------------------ utilities */

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("cev: error: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(2);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

typedef struct { char *p; size_t n, cap; } buf;

static void buf_put(buf *b, const void *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2 + 64;
        b->p = xrealloc(b->p, b->cap);
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void buf_puts(buf *b, const char *s) { buf_put(b, s, strlen(s)); }
static void buf_putc(buf *b, char c) { buf_put(b, &c, 1); }

static void buf_printf(buf *b, const char *fmt, ...) {
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n >= sizeof tmp) n = (int)sizeof tmp - 1;
    buf_put(b, tmp, (size_t)n);
}

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s (%s)", path, strerror(errno));
    if (fseek(f, 0, SEEK_END)) die("cannot seek %s", path);
    long sz = ftell(f);
    if (sz < 0) die("cannot size %s", path);
    rewind(f);
    unsigned char *d = xmalloc((size_t)sz + 1);
    if (fread(d, 1, (size_t)sz, f) != (size_t)sz) die("short read on %s", path);
    fclose(f);
    d[sz] = 0;
    *len = (size_t)sz;
    return d;
}

/* ------------------------------------------------------------------ SHA-256 (FIPS 180-4) */

typedef struct { uint32_t h[8]; uint64_t len; unsigned char blk[64]; size_t blen; } sha256_ctx;

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_block(sha256_ctx *c, const unsigned char *p) {
    uint32_t w[64], a, b, cc, d, e, f, g, h;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) | ((uint32_t)p[4*i+2] << 8) | p[4*i+3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i-15], 7) ^ ROR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROR(w[i-2], 17) ^ ROR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a = c->h[0]; b = c->h[1]; cc = c->h[2]; d = c->h[3]; e = c->h[4]; f = c->h[5]; g = c->h[6]; h = c->h[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1 = ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22);
        uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d; c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += h;
}

static void sha256_init(sha256_ctx *c) {
    static const uint32_t h0[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(c->h, h0, sizeof h0);
    c->len = 0; c->blen = 0;
}

static void sha256_update(sha256_ctx *c, const void *data, size_t n) {
    const unsigned char *p = data;
    c->len += n;
    while (n > 0) {
        size_t take = 64 - c->blen;
        if (take > n) take = n;
        memcpy(c->blk + c->blen, p, take);
        c->blen += take; p += take; n -= take;
        if (c->blen == 64) { sha256_block(c, c->blk); c->blen = 0; }
    }
}

static void sha256_final(sha256_ctx *c, char hex[65]) {
    uint64_t bits = c->len * 8;
    unsigned char pad = 0x80;
    sha256_update(c, &pad, 1);
    c->len -= 1;
    while (c->blen != 56) { unsigned char z = 0; sha256_update(c, &z, 1); c->len -= 1; }
    unsigned char lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (unsigned char)(bits >> (56 - 8 * i));
    sha256_update(c, lb, 8);
    for (int i = 0; i < 8; i++) snprintf(hex + 8 * i, 9, "%08x", c->h[i]);
    hex[64] = 0;
}

static void sha256_hex(const void *data, size_t n, char hex[65]) {
    sha256_ctx c; sha256_init(&c); sha256_update(&c, data, n); sha256_final(&c, hex);
}

static double parse_double_checked(const char *tok);

/* ------------------------------------------------------------------ JSON (RFC 8259) */

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jv {
    jtype t;
    int b;                 /* J_BOOL */
    char *tok;             /* J_NUM: the token text */
    double num;            /* J_NUM */
    int isint;             /* J_NUM: no '.', 'e', 'E' in the token */
    char *s; size_t slen;  /* J_STR: decoded UTF-8 (may contain NUL) */
    struct jv **items;     /* J_ARR / J_OBJ values */
    char **keys; size_t *keylens; /* J_OBJ keys (decoded UTF-8) */
    size_t n;
} jv;

typedef struct { const char *s; size_t i, n; const char *what; } jparser;

static void jerr(jparser *p, const char *msg) { die("%s: JSON parse error at byte %zu: %s", p->what, p->i, msg); }

static void jws(jparser *p) { while (p->i < p->n && (p->s[p->i] == ' ' || p->s[p->i] == '\t' || p->s[p->i] == '\n' || p->s[p->i] == '\r')) p->i++; }

static jv *jnew(jtype t) { jv *v = xmalloc(sizeof *v); memset(v, 0, sizeof *v); v->t = t; return v; }

static void put_utf8(buf *b, uint32_t cp) {
    unsigned char u[4];
    if (cp < 0x80) { u[0] = (unsigned char)cp; buf_put(b, u, 1); }
    else if (cp < 0x800) { u[0] = 0xC0 | (cp >> 6); u[1] = 0x80 | (cp & 0x3F); buf_put(b, u, 2); }
    else if (cp < 0x10000) { u[0] = 0xE0 | (cp >> 12); u[1] = 0x80 | ((cp >> 6) & 0x3F); u[2] = 0x80 | (cp & 0x3F); buf_put(b, u, 3); }
    else { u[0] = 0xF0 | (cp >> 18); u[1] = 0x80 | ((cp >> 12) & 0x3F); u[2] = 0x80 | ((cp >> 6) & 0x3F); u[3] = 0x80 | (cp & 0x3F); buf_put(b, u, 4); }
}

static int hex4(jparser *p, size_t at) {
    int v = 0;
    for (int k = 0; k < 4; k++) {
        if (at + k >= p->n) jerr(p, "truncated \\u escape");
        char c = p->s[at + k]; v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else jerr(p, "bad \\u escape");
    }
    return v;
}

static void jstring(jparser *p, char **out, size_t *outlen) {
    buf b = {0};
    if (p->s[p->i] != '"') jerr(p, "expected string");
    p->i++;
    for (;;) {
        if (p->i >= p->n) jerr(p, "unterminated string");
        unsigned char c = (unsigned char)p->s[p->i++];
        if (c == '"') break;
        if (c == '\\') {
            if (p->i >= p->n) jerr(p, "bad escape");
            char e = p->s[p->i++];
            switch (e) {
            case '"': buf_putc(&b, '"'); break;
            case '\\': buf_putc(&b, '\\'); break;
            case '/': buf_putc(&b, '/'); break;
            case 'b': buf_putc(&b, '\b'); break;
            case 'f': buf_putc(&b, '\f'); break;
            case 'n': buf_putc(&b, '\n'); break;
            case 'r': buf_putc(&b, '\r'); break;
            case 't': buf_putc(&b, '\t'); break;
            case 'u': {
                uint32_t cp = (uint32_t)hex4(p, p->i); p->i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && p->i + 1 < p->n && p->s[p->i] == '\\' && p->s[p->i+1] == 'u') {
                    uint32_t lo = (uint32_t)hex4(p, p->i + 2);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); p->i += 6; }
                }
                put_utf8(&b, cp);
                break; }
            default: jerr(p, "bad escape character");
            }
        } else if (c < 0x20) {
            jerr(p, "control character in string");
        } else {
            buf_put(&b, &c, 1);
        }
    }
    if (!b.p) buf_put(&b, "", 0);
    *out = b.p; *outlen = b.n;
}

static jv *jvalue(jparser *p);

static jv *jvalue(jparser *p) {
    jws(p);
    if (p->i >= p->n) jerr(p, "unexpected end");
    char c = p->s[p->i];
    if (c == '{') {
        jv *v = jnew(J_OBJ); size_t cap = 0;
        p->i++; jws(p);
        if (p->i < p->n && p->s[p->i] == '}') { p->i++; return v; }
        for (;;) {
            jws(p);
            char *k; size_t kl; jstring(p, &k, &kl);
            jws(p);
            if (p->i >= p->n || p->s[p->i] != ':') jerr(p, "expected ':'");
            p->i++;
            jv *val = jvalue(p);
            if (v->n == cap) { cap = cap * 2 + 8; v->items = xrealloc(v->items, cap * sizeof *v->items);
                               v->keys = xrealloc(v->keys, cap * sizeof *v->keys); v->keylens = xrealloc(v->keylens, cap * sizeof *v->keylens); }
            v->items[v->n] = val; v->keys[v->n] = k; v->keylens[v->n] = kl; v->n++;
            jws(p);
            if (p->i < p->n && p->s[p->i] == ',') { p->i++; continue; }
            if (p->i < p->n && p->s[p->i] == '}') { p->i++; return v; }
            jerr(p, "expected ',' or '}'");
        }
    }
    if (c == '[') {
        jv *v = jnew(J_ARR); size_t cap = 0;
        p->i++; jws(p);
        if (p->i < p->n && p->s[p->i] == ']') { p->i++; return v; }
        for (;;) {
            jv *val = jvalue(p);
            if (v->n == cap) { cap = cap * 2 + 8; v->items = xrealloc(v->items, cap * sizeof *v->items); }
            v->items[v->n++] = val;
            jws(p);
            if (p->i < p->n && p->s[p->i] == ',') { p->i++; continue; }
            if (p->i < p->n && p->s[p->i] == ']') { p->i++; return v; }
            jerr(p, "expected ',' or ']'");
        }
    }
    if (c == '"') { jv *v = jnew(J_STR); jstring(p, &v->s, &v->slen); return v; }
    if (!strncmp(p->s + p->i, "null", 4)) { p->i += 4; return jnew(J_NULL); }
    if (!strncmp(p->s + p->i, "true", 4)) { p->i += 4; jv *v = jnew(J_BOOL); v->b = 1; return v; }
    if (!strncmp(p->s + p->i, "false", 5)) { p->i += 5; jv *v = jnew(J_BOOL); v->b = 0; return v; }
    if (c == '-' || (c >= '0' && c <= '9')) {
        size_t st = p->i; int isint = 1;
        if (p->s[p->i] == '-') p->i++;
        while (p->i < p->n) {
            char d = p->s[p->i];
            if (d >= '0' && d <= '9') { p->i++; continue; }
            if (d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') { isint = 0; p->i++; continue; }
            break;
        }
        jv *v = jnew(J_NUM);
        v->tok = xmalloc(p->i - st + 1); memcpy(v->tok, p->s + st, p->i - st); v->tok[p->i - st] = 0;
        v->isint = isint;
        v->num = parse_double_checked(v->tok);
        return v;
    }
    jerr(p, "unexpected character");
    return NULL;
}

static jv *jparse(const char *text, size_t n, const char *what) {
    jparser p = { text, 0, n, what };
    jv *v = jvalue(&p);
    jws(&p);
    if (p.i != p.n) jerr(&p, "trailing data");
    return v;
}

/* lookups: NULL when absent; jget on a non-object is NULL */
static jv *jget(jv *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    size_t kl = strlen(key);
    for (size_t i = 0; i < o->n; i++)
        if (o->keylens[i] == kl && !memcmp(o->keys[i], key, kl)) return o->items[i];
    return NULL;
}
static jv *jidx(jv *a, size_t i) { return (a && a->t == J_ARR && i < a->n) ? a->items[i] : NULL; }
static int jisnull(jv *v) { return !v || v->t == J_NULL; }
static double jnum(jv *v, double dflt) { return (v && v->t == J_NUM) ? v->num : dflt; }
static const char *jstr(jv *v) { return (v && v->t == J_STR) ? v->s : NULL; }
static int jstreq(jv *v, const char *s) { return v && v->t == J_STR && v->slen == strlen(s) && !memcmp(v->s, s, v->slen); }
static int jbool(jv *v, int dflt) { return (v && v->t == J_BOOL) ? v->b : dflt; }
static size_t jlen(jv *v) { return (v && (v->t == J_ARR || v->t == J_OBJ)) ? v->n : 0; }


/* ------------------------------------------------------------------ exact decimal <-> binary64
 * Python's float repr is the shortest decimal that round-trips, the closest such decimal when several exist
 * (ties to even). Reproducing it bit-for-bit needs correctly rounded conversions, which the C library is not
 * guaranteed to supply (ties in printf, hard cases in strtod). Both directions are therefore done here with
 * exact integer arithmetic: a double M 2^E is expanded to its exact decimal digits, candidates are rounded
 * half-even to p digits and tested against the exact rounding interval of the double; a decimal token is
 * parsed with strtod and the result checked (and nudged) against the same interval. */

#define BIG_LIMBS 192
typedef struct { uint32_t l[BIG_LIMBS]; int n; } big;      /* base 1e9, little-endian */

static void big_from_u64(big *b, uint64_t v) {
    b->n = 0;
    while (v) { if (b->n >= BIG_LIMBS) die("bignum overflow"); b->l[b->n++] = (uint32_t)(v % 1000000000u); v /= 1000000000u; }
}
static void big_mul_small(big *b, uint32_t m) {
    uint64_t carry = 0;
    for (int i = 0; i < b->n; i++) { uint64_t t = (uint64_t)b->l[i] * m + carry; b->l[i] = (uint32_t)(t % 1000000000u); carry = t / 1000000000u; }
    while (carry) { if (b->n >= BIG_LIMBS) die("bignum overflow"); b->l[b->n++] = (uint32_t)(carry % 1000000000u); carry /= 1000000000u; }
}
static void big_mul_pow(big *b, int base, int e) {      /* base 2, 5 or 10 */
    uint32_t chunk = 1; int per = 0;
    while (chunk * (uint64_t)base < 4000000000u) { chunk *= (uint32_t)base; per++; }
    while (e >= per) { big_mul_small(b, chunk); e -= per; }
    uint32_t rest = 1; while (e-- > 0) rest *= (uint32_t)base;
    if (rest > 1) big_mul_small(b, rest);
}
static int big_cmp(const big *a, const big *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (int i = a->n - 1; i >= 0; i--) if (a->l[i] != b->l[i]) return a->l[i] < b->l[i] ? -1 : 1;
    return 0;
}
static void big_from_digits(big *b, const char *d, int nd) {
    b->n = 0;
    for (int i = 0; i < nd; i++) { big_mul_small(b, 10); if (d[i] != '0') { big t; big_from_u64(&t, (uint64_t)(d[i] - '0')); /* add */
        uint64_t carry = t.n ? t.l[0] : 0; for (int j = 0; j < b->n && carry; j++) { uint64_t s = b->l[j] + carry; b->l[j] = (uint32_t)(s % 1000000000u); carry = s / 1000000000u; }
        if (carry) { if (b->n >= BIG_LIMBS) die("bignum overflow"); b->l[b->n++] = (uint32_t)carry; } } }
}
static int big_to_dec(const big *b, char *out) {   /* returns length; "0" for zero */
    if (b->n == 0) { out[0] = '0'; out[1] = 0; return 1; }
    int L = sprintf(out, "%u", b->l[b->n - 1]);
    for (int i = b->n - 2; i >= 0; i--) L += sprintf(out + L, "%09u", b->l[i]);
    return L;
}

/* decomposition x = M 2^E (x > 0, finite) */
static void decompose(double x, uint64_t *M, int *E, int *pow2_boundary) {
    uint64_t bits; memcpy(&bits, &x, 8);
    uint64_t mant = bits & ((1ull << 52) - 1); int ex = (int)((bits >> 52) & 0x7FF);
    if (ex == 0) { *M = mant; *E = -1074; *pow2_boundary = 0; }
    else { *M = mant | (1ull << 52); *E = ex - 1075; *pow2_boundary = (mant == 0 && ex > 1); }
}

/* does the decimal d (nd digits) * 10^t lie in the rounding interval of M 2^E?  side: -1 lower, +1 upper */
static int within_bound(const char *d, int nd, int t, uint64_t M, int E, int pow2_boundary, int side) {
    /* bound = K 2^(E-2), K = 4M+2 (upper), 4M-2 (lower), 4M-1 (lower at a power-of-two boundary) */
    uint64_t K = side > 0 ? 4 * M + 2 : (pow2_boundary ? 4 * M - 1 : 4 * M - 2);
    int e2 = E - 2;
    int a = t < 0 ? -t : 0, b = e2 < 0 ? -e2 : 0;
    big L, R;
    big_from_digits(&L, d, nd); big_mul_pow(&L, 10, t + a); big_mul_pow(&L, 2, b);
    big_from_u64(&R, K); big_mul_pow(&R, 2, e2 + b); big_mul_pow(&R, 10, a);
    int c = big_cmp(&L, &R);
    int inclusive = (M % 2 == 0);
    if (side > 0) return inclusive ? c <= 0 : c < 0;
    return inclusive ? c >= 0 : c > 0;
}

/* shortest round-trip digits of x > 0: digits (no trailing zeros) and decpt with x = 0.d1d2... * 10^decpt */
static void shortest_digits(double x, char *digits, int *decpt) {
    uint64_t M; int E, p2; decompose(x, &M, &E, &p2);
    big D; big_from_u64(&D, M);
    int s = 0;
    if (E >= 0) big_mul_pow(&D, 2, E); else { s = -E; big_mul_pow(&D, 5, s); }
    char dstr[1200]; int L = big_to_dec(&D, dstr);
    int exact_decpt = L - s;
    for (int p = 1; p <= 17; p++) {
        char cand[40]; int nd = p < L ? p : L; memcpy(cand, dstr, (size_t)nd); cand[nd] = 0; int cdecpt = exact_decpt;
        if (p < L) {
            int next = dstr[p] - '0', rest_nonzero = 0;
            for (int i = p + 1; i < L; i++) if (dstr[i] != '0') { rest_nonzero = 1; break; }
            int round_up = next > 5 || (next == 5 && (rest_nonzero || ((cand[nd - 1] - '0') % 2 == 1)));
            if (round_up) {
                int i = nd - 1;
                while (i >= 0 && cand[i] == '9') { cand[i] = '0'; i--; }
                if (i < 0) { memmove(cand + 1, cand, (size_t)nd + 1); cand[0] = '1'; nd++; cdecpt++; }
                else cand[i]++;
            }
        }
        int n2 = nd; while (n2 > 1 && cand[n2 - 1] == '0') n2--;
        int t = cdecpt - n2;
        if (within_bound(cand, n2, t, M, E, p2, -1) && within_bound(cand, n2, t, M, E, p2, +1)) {
            memcpy(digits, cand, (size_t)n2); digits[n2] = 0; *decpt = cdecpt; return;
        }
    }
    die("shortest digits not found for %.17g", x);
}

/* correctly rounded parse of a JSON number token (checked with the exact interval test) */
static double parse_double_checked(const char *tok) {
    double d = strtod(tok, NULL);
    if (!isfinite(d)) return d;
    /* decimal digits and exponent of the token */
    const char *s = tok; int neg = 0; if (*s == '-') { neg = 1; s++; }
    char dig[400]; int nd = 0, t = 0, seen_point = 0, lead = 1;
    for (; *s && *s != 'e' && *s != 'E'; s++) {
        if (*s == '.') { seen_point = 1; continue; }
        if (*s < '0' || *s > '9') return d;
        if (lead && *s == '0') { if (seen_point) t--; continue; }
        lead = 0;
        if (nd < 380) { dig[nd++] = *s; if (seen_point) t--; } else if (!seen_point) t++;
    }
    if (*s == 'e' || *s == 'E') t += atoi(s + 1);
    if (nd == 0) return neg ? -0.0 : 0.0;
    double ax = fabs(d);
    if (ax == 0.0) return d;      /* a nonzero token that parsed to zero: below the subnormal range; left as is */
    for (int tries = 0; tries < 8; tries++) {
        uint64_t M; int E, p2; decompose(ax, &M, &E, &p2);
        int lo_ok = within_bound(dig, nd, t, M, E, p2, -1), hi_ok = within_bound(dig, nd, t, M, E, p2, +1);
        if (lo_ok && hi_ok) break;
        ax = lo_ok ? nextafter(ax, INFINITY) : nextafter(ax, 0.0);
    }
    return neg ? -ax : ax;
}

/* ------------------------------------------------------------------ canonical JSON (certify-ed-canonical-json-1)
 * json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False): keys sorted by
 * code point, floats as Python repr (shortest round-trip), integers verbatim, strings with only ", \ and
 * control characters escaped (\uXXXX lower-case hex for controls without a short form). */

static void py_float_repr(double x, char *out) {
    if (x != x || x == INFINITY || x == -INFINITY) die("non-finite float in the JSON body (allow_nan=False)");
    if (x == 0.0) { strcpy(out, signbit(x) ? "-0.0" : "0.0"); return; }
    char digits[40]; int decpt, nd;
    shortest_digits(fabs(x), digits, &decpt);
    nd = (int)strlen(digits);
    char *o = out;
    if (x < 0) *o++ = '-';
    if (decpt <= -4 || decpt > 16) {
        *o++ = digits[0];
        if (nd > 1) { *o++ = '.'; memcpy(o, digits + 1, (size_t)nd - 1); o += nd - 1; }
        int ex = decpt - 1;
        o += sprintf(o, "e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    } else if (decpt <= 0) {
        *o++ = '0'; *o++ = '.';
        for (int k = 0; k < -decpt; k++) *o++ = '0';
        memcpy(o, digits, (size_t)nd); o += nd;
    } else if (decpt >= nd) {
        memcpy(o, digits, (size_t)nd); o += nd;
        for (int k = 0; k < decpt - nd; k++) *o++ = '0';
        *o++ = '.'; *o++ = '0';
    } else {
        memcpy(o, digits, (size_t)decpt); o += decpt;
        *o++ = '.';
        memcpy(o, digits + decpt, (size_t)(nd - decpt)); o += nd - decpt;
    }
    *o = 0;
}

static void canon_string(buf *b, const char *s, size_t n) {
    buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\n': buf_puts(b, "\\n"); break;
        case '\r': buf_puts(b, "\\r"); break;
        case '\t': buf_puts(b, "\\t"); break;
        case '\b': buf_puts(b, "\\b"); break;
        case '\f': buf_puts(b, "\\f"); break;
        default:
            if (c < 0x20) buf_printf(b, "\\u%04x", c);
            else buf_put(b, &c, 1);
        }
    }
    buf_putc(b, '"');
}

static jv *g_sort_obj;
static int keycmp(const void *a, const void *b) {
    size_t ia = *(const size_t *)a, ib = *(const size_t *)b;
    size_t la = g_sort_obj->keylens[ia], lb = g_sort_obj->keylens[ib];
    int c = memcmp(g_sort_obj->keys[ia], g_sort_obj->keys[ib], la < lb ? la : lb);
    if (c) return c;
    return la < lb ? -1 : (la > lb ? 1 : 0);
}

/* skip: a top-level key to omit (the integrity block), or NULL */
static void canon(buf *b, jv *v, const char *skip) {
    switch (v->t) {
    case J_NULL: buf_puts(b, "null"); break;
    case J_BOOL: buf_puts(b, v->b ? "true" : "false"); break;
    case J_NUM:
        if (v->isint) buf_puts(b, v->tok);
        else { char t[48]; py_float_repr(v->num, t); buf_puts(b, t); }
        break;
    case J_STR: canon_string(b, v->s, v->slen); break;
    case J_ARR:
        buf_putc(b, '[');
        for (size_t i = 0; i < v->n; i++) { if (i) buf_putc(b, ','); canon(b, v->items[i], NULL); }
        buf_putc(b, ']');
        break;
    case J_OBJ: {
        size_t *idx = xmalloc((v->n + 1) * sizeof *idx);
        for (size_t i = 0; i < v->n; i++) idx[i] = i;
        g_sort_obj = v;
        qsort(idx, v->n, sizeof *idx, keycmp);
        buf_putc(b, '{');
        int first = 1;
        for (size_t i = 0; i < v->n; i++) {
            size_t k = idx[i];
            if (skip && v->keylens[k] == strlen(skip) && !memcmp(v->keys[k], skip, v->keylens[k])) continue;
            if (!first) buf_putc(b, ',');
            first = 0;
            canon_string(b, v->keys[k], v->keylens[k]);
            buf_putc(b, ':');
            canon(b, v->items[k], NULL);
        }
        buf_putc(b, '}');
        free(idx);
        break; }
    }
}


/* ------------------------------------------------------------------ DEFLATE (RFC 1951) decoder, for
 * savez_compressed / scipy.sparse.save_npz members (method 8). A canonical-Huffman decoder in the style of
 * zlib's puff: stored, fixed and dynamic blocks; no error recovery beyond rejecting malformed input. */

typedef struct { const unsigned char *in; size_t inlen, inpos; unsigned int bitbuf; int bitcnt; buf *out; } infl;

static int inf_bits(infl *s, int need) {
    unsigned int val = s->bitbuf;
    while (s->bitcnt < need) {
        if (s->inpos >= s->inlen) die("deflate: input exhausted");
        val |= (unsigned int)s->in[s->inpos++] << s->bitcnt;
        s->bitcnt += 8;
    }
    s->bitbuf = val >> need; s->bitcnt -= need;
    return (int)(val & ((1u << need) - 1));
}

typedef struct { short count[16]; short symbol[290]; } huff;

static int inf_build(huff *h, const short *length, int n) {
    short offs[16]; int left = 1;
    for (int len = 0; len < 16; len++) h->count[len] = 0;
    for (int sym = 0; sym < n; sym++) h->count[length[sym]]++;
    if (h->count[0] == n) return 0;
    for (int len = 1; len < 16; len++) { left <<= 1; left -= h->count[len]; if (left < 0) return -1; }
    offs[1] = 0;
    for (int len = 1; len < 15; len++) offs[len + 1] = (short)(offs[len] + h->count[len]);
    for (int sym = 0; sym < n; sym++) if (length[sym]) h->symbol[offs[length[sym]]++] = (short)sym;
    return left;
}

static int inf_decode(infl *s, const huff *h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; len++) {
        code |= inf_bits(s, 1);
        int count = h->count[len];
        if (code - count < first) return h->symbol[index + (code - first)];
        index += count; first += count; first <<= 1; code <<= 1;
    }
    die("deflate: bad code");
    return -1;
}

static const short LBASE[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
static const short LEXT[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
static const short DBASE[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
static const short DEXT[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

static void inf_codes(infl *s, const huff *lencode, const huff *distcode) {
    for (;;) {
        int sym = inf_decode(s, lencode);
        if (sym < 256) { unsigned char c = (unsigned char)sym; buf_put(s->out, &c, 1); }
        else if (sym == 256) return;
        else {
            sym -= 257;
            if (sym >= 29) die("deflate: bad length symbol");
            int len = LBASE[sym] + inf_bits(s, LEXT[sym]);
            int dsym = inf_decode(s, distcode);
            if (dsym >= 30) die("deflate: bad distance symbol");
            size_t dist = (size_t)DBASE[dsym] + (size_t)inf_bits(s, DEXT[dsym]);
            if (dist > s->out->n) die("deflate: distance too far back");
            for (int i = 0; i < len; i++) { unsigned char c = (unsigned char)s->out->p[s->out->n - dist]; buf_put(s->out, &c, 1); }
        }
    }
}

static void inflate_raw(const unsigned char *in, size_t inlen, buf *out) {
    infl s = { in, inlen, 0, 0, 0, out };
    int last;
    do {
        last = inf_bits(&s, 1);
        int type = inf_bits(&s, 2);
        if (type == 0) {
            s.bitbuf = 0; s.bitcnt = 0;
            if (s.inpos + 4 > s.inlen) die("deflate: truncated stored block");
            unsigned int len = s.in[s.inpos] | (s.in[s.inpos + 1] << 8);
            unsigned int nlen = s.in[s.inpos + 2] | (s.in[s.inpos + 3] << 8);
            if ((len ^ 0xFFFF) != nlen) die("deflate: stored block length check failed");
            s.inpos += 4;
            if (s.inpos + len > s.inlen) die("deflate: truncated stored block");
            buf_put(out, s.in + s.inpos, len); s.inpos += len;
        } else if (type == 1) {
            static huff lencode, distcode; static int built = 0;
            if (!built) {
                short lengths[290]; int sym = 0;
                for (; sym < 144; sym++) lengths[sym] = 8;
                for (; sym < 256; sym++) lengths[sym] = 9;
                for (; sym < 280; sym++) lengths[sym] = 7;
                for (; sym < 288; sym++) lengths[sym] = 8;
                inf_build(&lencode, lengths, 288);
                for (sym = 0; sym < 30; sym++) lengths[sym] = 5;
                inf_build(&distcode, lengths, 30);
                built = 1;
            }
            inf_codes(&s, &lencode, &distcode);
        } else if (type == 2) {
            static const short order[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
            short lengths[320]; huff lencode, distcode;
            int nlen = inf_bits(&s, 5) + 257, ndist = inf_bits(&s, 5) + 1, ncode = inf_bits(&s, 4) + 4;
            if (nlen > 286 || ndist > 30) die("deflate: bad counts");
            int idx;
            for (idx = 0; idx < ncode; idx++) lengths[order[idx]] = (short)inf_bits(&s, 3);
            for (; idx < 19; idx++) lengths[order[idx]] = 0;
            if (inf_build(&lencode, lengths, 19) != 0) die("deflate: bad code lengths code");
            idx = 0;
            while (idx < nlen + ndist) {
                int sym = inf_decode(&s, &lencode);
                if (sym < 16) lengths[idx++] = (short)sym;
                else {
                    int len = 0, rep;
                    if (sym == 16) { if (idx == 0) die("deflate: repeat with no first length"); len = lengths[idx - 1]; rep = 3 + inf_bits(&s, 2); }
                    else if (sym == 17) rep = 3 + inf_bits(&s, 3);
                    else rep = 11 + inf_bits(&s, 7);
                    if (idx + rep > nlen + ndist) die("deflate: too many lengths");
                    while (rep--) lengths[idx++] = (short)len;
                }
            }
            if (lengths[256] == 0) die("deflate: no end-of-block code");
            int e1 = inf_build(&lencode, lengths, nlen);
            if (e1 < 0 || (e1 > 0 && nlen - lencode.count[0] != 1)) die("deflate: incomplete length code");
            int e2 = inf_build(&distcode, lengths + nlen, ndist);
            if (e2 < 0 || (e2 > 0 && ndist - distcode.count[0] != 1)) die("deflate: incomplete distance code");
            inf_codes(&s, &lencode, &distcode);
        } else die("deflate: invalid block type");
    } while (!last);
}

/* ------------------------------------------------------------------ NPZ / NPY reader
 * np.savez writes a ZIP archive (method 0, stored) of .npy members, one per array. The central directory is
 * read (EOCD, zip64 EOCD when present), then every member's local header; compressed members (savez_compressed)
 * are inflated with the decoder above. .npy: magic, version 1/2/3, header dict with descr, fortran_order, shape. */

typedef struct {
    char name[256];
    char descr[16];     /* e.g. "<f8", "<i4", "|S3" */
    int fortran;
    int ndim;
    size_t shape[4];
    size_t itemsize;
    size_t count;       /* product of the shape */
    unsigned char *data;
    size_t nbytes;
} nparr;

static uint16_t rd16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const unsigned char *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t rd64(const unsigned char *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

static size_t descr_itemsize(const char *d) {
    const char *q = d + 1;
    while (*q && (*q < '0' || *q > '9')) q++;
    return (size_t)strtoul(q, NULL, 10);
}

static void parse_npy(const unsigned char *m, size_t n, nparr *a, const char *name) {
    if (n < 10 || memcmp(m, "\x93NUMPY", 6)) die("%s: not a .npy member", name);
    int major = m[6];
    size_t hl, off;
    if (major == 1) { hl = rd16(m + 8); off = 10; }
    else if (major == 2 || major == 3) { hl = rd32(m + 8); off = 12; }
    else die("%s: unsupported .npy version %d", name, major);
    if (off + hl > n) die("%s: truncated .npy header", name);
    char *h = xmalloc(hl + 1); memcpy(h, m + off, hl); h[hl] = 0;
    char *d = strstr(h, "'descr'");
    if (!d) die("%s: .npy header without descr", name);
    d = strchr(d + 7, '\''); if (!d) die("%s: bad descr", name);
    char *e = strchr(d + 1, '\''); if (!e || (size_t)(e - d - 1) >= sizeof a->descr) die("%s: bad descr", name);
    memcpy(a->descr, d + 1, (size_t)(e - d - 1)); a->descr[e - d - 1] = 0;
    char *f = strstr(h, "'fortran_order'");
    if (!f) die("%s: .npy header without fortran_order", name);
    { char *t = strstr(f + 15, "True"), *fa = strstr(f + 15, "False");
      a->fortran = t && (!fa || t < fa); }
    char *s = strstr(h, "'shape'");
    if (!s) die("%s: .npy header without shape", name);
    s = strchr(s, '('); if (!s) die("%s: bad shape", name);
    s++;
    a->ndim = 0; a->count = 1;
    while (*s && *s != ')') {
        while (*s == ' ' || *s == ',') s++;
        if (*s == ')') break;
        if (*s < '0' || *s > '9') die("%s: bad shape", name);
        if (a->ndim >= 4) die("%s: more than 4 dimensions", name);
        a->shape[a->ndim++] = (size_t)strtoull(s, &s, 10);
    }
    for (int i = 0; i < a->ndim; i++) a->count *= a->shape[i];
    a->itemsize = descr_itemsize(a->descr);
    a->nbytes = a->count * a->itemsize;
    if (off + hl + a->nbytes > n) die("%s: .npy data truncated (%zu bytes expected)", name, a->nbytes);
    a->data = xmalloc(a->nbytes + 1);
    memcpy(a->data, m + off + hl, a->nbytes);
    snprintf(a->name, sizeof a->name, "%s", name);
    free(h);
}

static int load_npz(const char *path, nparr **out, int *nout) {
    size_t n; unsigned char *z = read_file(path, &n);
    if (n < 22) die("%s: not a zip archive", path);
    /* EOCD */
    size_t i = n - 22, lim = n > 22 + 65535 ? n - 22 - 65535 : 0;
    for (;; i--) {
        if (rd32(z + i) == 0x06054b50) break;
        if (i == lim) die("%s: zip end-of-central-directory not found", path);
    }
    uint64_t nent = rd16(z + i + 10), cdsize = rd32(z + i + 12), cdoff = rd32(z + i + 16);
    if (nent == 0xFFFF || cdsize == 0xFFFFFFFFu || cdoff == 0xFFFFFFFFu) {
        if (i < 20 || rd32(z + i - 20) != 0x07064b50) die("%s: zip64 locator missing", path);
        uint64_t z64 = rd64(z + i - 20 + 8);
        if (z64 + 56 > n || rd32(z + z64) != 0x06064b50) die("%s: zip64 EOCD missing", path);
        nent = rd64(z + z64 + 32); cdsize = rd64(z + z64 + 40); cdoff = rd64(z + z64 + 48);
    }
    if (cdoff + cdsize > n) die("%s: central directory out of range", path);
    nparr *arrs = xmalloc((size_t)(nent + 1) * sizeof *arrs);
    memset(arrs, 0, (size_t)(nent + 1) * sizeof *arrs);
    size_t p = (size_t)cdoff; int k = 0;
    for (uint64_t e = 0; e < nent; e++) {
        if (p + 46 > n || rd32(z + p) != 0x02014b50) die("%s: bad central directory entry", path);
        uint16_t method = rd16(z + p + 10);
        uint64_t csize = rd32(z + p + 20), usize = rd32(z + p + 24);
        uint16_t nl = rd16(z + p + 28), xl = rd16(z + p + 30), cl = rd16(z + p + 32);
        uint64_t loff = rd32(z + p + 42);
        char name[256]; size_t cp = nl < 255 ? nl : 255; memcpy(name, z + p + 46, cp); name[cp] = 0;
        /* zip64 extra in the central entry: fields in the order usize, csize, loff, only those == 0xFFFFFFFF */
        size_t xp = p + 46 + nl, xend = xp + xl;
        while (xp + 4 <= xend) {
            uint16_t id = rd16(z + xp), sz = rd16(z + xp + 2); size_t q = xp + 4;
            if (id == 1) {
                if (usize == 0xFFFFFFFFu && q + 8 <= xend) { usize = rd64(z + q); q += 8; }
                if (csize == 0xFFFFFFFFu && q + 8 <= xend) { csize = rd64(z + q); q += 8; }
                if (loff == 0xFFFFFFFFu && q + 8 <= xend) { loff = rd64(z + q); q += 8; }
            }
            xp += 4 + sz;
        }
        if (method != 0 && method != 8) die("%s: member %s uses compression method %u (only stored and deflate are supported)", path, name, method);
        if (loff + 30 > n || rd32(z + loff) != 0x04034b50) die("%s: bad local header for %s", path, name);
        uint16_t lnl = rd16(z + loff + 26), lxl = rd16(z + loff + 28);
        size_t dstart = (size_t)loff + 30 + lnl + lxl;
        if (dstart + csize > n) die("%s: member %s data out of range", path, name);
        size_t L = strlen(name);
        if (L > 4 && !strcmp(name + L - 4, ".npy")) name[L - 4] = 0;
        if (method == 8) {
            buf ob = {0};
            inflate_raw(z + dstart, (size_t)csize, &ob);
            if (ob.n != usize) die("%s: member %s inflated to %zu bytes, %llu expected", path, name, ob.n, (unsigned long long)usize);
            parse_npy((const unsigned char *)ob.p, ob.n, &arrs[k], name);
            free(ob.p);
        } else {
            parse_npy(z + dstart, (size_t)usize, &arrs[k], name);
        }
        k++;
        p += 46 + nl + xl + cl;
    }
    free(z);
    *out = arrs; *nout = k;
    return k;
}

static nparr *np_find(nparr *a, int n, const char *name) {
    for (int i = 0; i < n; i++) if (!strcmp(a[i].name, name)) return &a[i];
    return NULL;
}

/* SHA-256 of dtype.str + json.dumps(list(shape)) (default separators ", ") + C-order bytes, as certify-ed's
 * array_sha256. Fortran-ordered 2-D members are transposed into C order first. */
static void array_hash(const nparr *a, char hex[65]) {
    sha256_ctx c; sha256_init(&c);
    sha256_update(&c, a->descr, strlen(a->descr));
    char sh[128]; size_t o = 0;
    sh[o++] = '[';
    for (int i = 0; i < a->ndim; i++) o += (size_t)snprintf(sh + o, sizeof sh - o, "%s%zu", i ? ", " : "", a->shape[i]);
    sh[o++] = ']'; sh[o] = 0;
    sha256_update(&c, sh, o);
    if (a->fortran && a->ndim == 2) {
        size_t r = a->shape[0], cc = a->shape[1], it = a->itemsize;
        unsigned char *t = xmalloc(a->nbytes + 1);
        for (size_t i = 0; i < r; i++)
            for (size_t j = 0; j < cc; j++)
                memcpy(t + (i * cc + j) * it, a->data + (j * r + i) * it, it);
        sha256_update(&c, t, a->nbytes);
        free(t);
    } else if (a->fortran && a->ndim > 2) {
        die("%s: Fortran-ordered array with more than 2 dimensions is not supported", a->name);
    } else {
        sha256_update(&c, a->data, a->nbytes);
    }
    sha256_final(&c, hex);
}

/* read element i of a real array as double (float64 only), row-major element index */
static double f64_at(const nparr *a, size_t i) { double v; memcpy(&v, a->data + 8 * i, 8); return v; }
static int64_t int_at(const nparr *a, size_t i) {
    if (!strcmp(a->descr, "<i4")) { int32_t v; memcpy(&v, a->data + 4 * i, 4); return v; }
    if (!strcmp(a->descr, "<i8")) { int64_t v; memcpy(&v, a->data + 8 * i, 8); return v; }
    die("%s: integer array of dtype %s is not supported", a->name, a->descr);
    return 0;
}
/* column j, row i of a 2-D float64 array, honouring the storage order */
static double f64_2d(const nparr *a, size_t i, size_t j) {
    size_t idx = a->fortran ? j * a->shape[0] + i : i * a->shape[1] + j;
    return f64_at(a, idx);
}

/* ------------------------------------------------------------------ directed rounding
 * Each helper sets the rounding mode it needs and leaves it set; the operands are volatile so that the
 * compiler cannot fold or reorder the operation (-frounding-math is required as well). */

static double ru_add(double a, double b) { fesetround(FE_UPWARD); volatile double x = a, y = b; volatile double r = x + y; return r; }
static double ru_sub(double a, double b) { fesetround(FE_UPWARD); volatile double x = a, y = b; volatile double r = x - y; return r; }
static double rd_sub(double a, double b) { fesetround(FE_DOWNWARD); volatile double x = a, y = b; volatile double r = x - y; return r; }
static double ru_mul(double a, double b) { fesetround(FE_UPWARD); volatile double x = a, y = b; volatile double r = x * y; return r; }
static double ru_div(double a, double b) { fesetround(FE_UPWARD); volatile double x = a, y = b; volatile double r = x / y; return r; }
static double rd_div(double a, double b) { fesetround(FE_DOWNWARD); volatile double x = a, y = b; volatile double r = x / y; return r; }
static double rd_sqrt(double a) { fesetround(FE_DOWNWARD); volatile double x = a; volatile double r = sqrt(x); return r; }

/* interval helpers: [lo, hi] encloses the exact value */
typedef struct { double lo, hi; } ival;

/* |x| for x in [lo, hi] */
static ival iv_abs(ival a) {
    ival r;
    if (a.lo >= 0) { r = a; }
    else if (a.hi <= 0) { r.lo = -a.hi; r.hi = -a.lo; }
    else { r.lo = 0; r.hi = (-a.lo > a.hi) ? -a.lo : a.hi; }
    return r;
}

/* ------------------------------------------------------------------ check list */

typedef struct { char id[48]; char st[16]; char detail[700]; } check;
static check *CH; static int NCH, CHCAP;
static int n_pass, n_fail, n_skip, n_repro, n_norepro, n_info;

static void add(const char *st, const char *id, const char *fmt, ...) {
    if (NCH == CHCAP) { CHCAP = CHCAP * 2 + 32; CH = xrealloc(CH, (size_t)CHCAP * sizeof *CH); }
    check *c = &CH[NCH++];
    snprintf(c->id, sizeof c->id, "%s", id);
    snprintf(c->st, sizeof c->st, "%s", st);
    va_list ap; va_start(ap, fmt); vsnprintf(c->detail, sizeof c->detail, fmt, ap); va_end(ap);
    if (!strcmp(st, "PASS")) n_pass++;
    else if (!strcmp(st, "FAIL")) n_fail++;
    else if (!strcmp(st, "SKIP")) n_skip++;
    else if (!strcmp(st, "REPRODUCED")) n_repro++;
    else if (!strcmp(st, "NOT_REPRODUCED")) n_norepro++;
    else n_info++;
}
#define PASSFAIL(cond) ((cond) ? "PASS" : "FAIL")

/* ------------------------------------------------------------------ CSR operator (scipy.sparse.save_npz layout) */

typedef struct { size_t n; size_t nnz; int64_t *indptr; int64_t *indices; double *data; } csr;

static void csr_canonicalize(csr *A) {
    /* per row: sort by column, sum duplicates, drop explicit zeros */
    int64_t *ip = xmalloc((A->n + 1) * sizeof *ip);
    int64_t *ix = xmalloc((A->nnz + 1) * sizeof *ix);
    double *dt = xmalloc((A->nnz + 1) * sizeof *dt);
    size_t out = 0;
    fesetround(FE_TONEAREST);
    for (size_t r = 0; r < A->n; r++) {
        ip[r] = (int64_t)out;
        size_t a = (size_t)A->indptr[r], b = (size_t)A->indptr[r + 1];
        /* insertion sort of the row (rows are short) */
        for (size_t i = a + 1; i < b; i++) {
            int64_t ci = A->indices[i]; double vi = A->data[i]; size_t j = i;
            while (j > a && A->indices[j - 1] > ci) { A->indices[j] = A->indices[j - 1]; A->data[j] = A->data[j - 1]; j--; }
            A->indices[j] = ci; A->data[j] = vi;
        }
        for (size_t i = a; i < b;) {
            int64_t cidx = A->indices[i]; volatile double s = A->data[i]; size_t j = i + 1;
            while (j < b && A->indices[j] == cidx) { s = s + A->data[j]; j++; }
            if (s != 0.0) { ix[out] = cidx; dt[out] = s; out++; }
            i = j;
        }
    }
    ip[A->n] = (int64_t)out;
    free(A->indptr); free(A->indices); free(A->data);
    A->indptr = ip; A->indices = ix; A->data = dt; A->nnz = out;
}

static csr *csr_load(const char *path, const char **why) {
    nparr *arrs; int na; load_npz(path, &arrs, &na);
    nparr *fmt = np_find(arrs, na, "format"), *sh = np_find(arrs, na, "shape"), *dat = np_find(arrs, na, "data"),
          *ind = np_find(arrs, na, "indices"), *ptr = np_find(arrs, na, "indptr");
    if (!fmt || !sh || !dat || !ind || !ptr) { *why = "not a scipy save_npz file (keys format, shape, data, indices, indptr)"; return NULL; }
    if (fmt->nbytes < 3 || memcmp(fmt->data, "csr", 3)) { *why = "operator is not in CSR format"; return NULL; }
    if (strcmp(dat->descr, "<f8")) { *why = "operator data is not float64 (complex operators are not supported by this verifier)"; return NULL; }
    csr *A = xmalloc(sizeof *A);
    A->n = (size_t)int_at(sh, 0);
    if ((size_t)int_at(sh, 1) != A->n) { *why = "operator is not square"; return NULL; }
    A->nnz = dat->count;
    A->indptr = xmalloc((A->n + 1) * sizeof *A->indptr);
    A->indices = xmalloc((A->nnz + 1) * sizeof *A->indices);
    A->data = xmalloc((A->nnz + 1) * sizeof *A->data);
    for (size_t i = 0; i <= A->n; i++) A->indptr[i] = int_at(ptr, i);
    for (size_t i = 0; i < A->nnz; i++) { A->indices[i] = int_at(ind, i); A->data[i] = f64_at(dat, i); }
    csr_canonicalize(A);
    return A;
}

/* (A + A^T) / 2 with round-to-nearest, as certify-ed forms the Hermitian part of a non-symmetric producer matrix */
static csr *csr_hermitian_part(const csr *A) {
    size_t n = A->n;
    /* transpose */
    int64_t *cnt = xmalloc((n + 1) * sizeof *cnt); memset(cnt, 0, (n + 1) * sizeof *cnt);
    for (size_t i = 0; i < A->nnz; i++) cnt[A->indices[i] + 1]++;
    for (size_t r = 0; r < n; r++) cnt[r + 1] += cnt[r];
    int64_t *tix = xmalloc((A->nnz + 1) * sizeof *tix); double *tdt = xmalloc((A->nnz + 1) * sizeof *tdt);
    int64_t *pos = xmalloc((n + 1) * sizeof *pos); memcpy(pos, cnt, (n + 1) * sizeof *pos);
    for (size_t r = 0; r < n; r++)
        for (size_t i = (size_t)A->indptr[r]; i < (size_t)A->indptr[r + 1]; i++) {
            size_t c = (size_t)A->indices[i]; tix[pos[c]] = (int64_t)r; tdt[pos[c]] = A->data[i]; pos[c]++;
        }
    /* merge rows of A and A^T (both sorted by column) */
    csr *H = xmalloc(sizeof *H); H->n = n;
    H->indptr = xmalloc((n + 1) * sizeof *H->indptr);
    H->indices = xmalloc((2 * A->nnz + 1) * sizeof *H->indices);
    H->data = xmalloc((2 * A->nnz + 1) * sizeof *H->data);
    size_t out = 0;
    fesetround(FE_TONEAREST);
    for (size_t r = 0; r < n; r++) {
        H->indptr[r] = (int64_t)out;
        size_t i = (size_t)A->indptr[r], ie = (size_t)A->indptr[r + 1], j = (size_t)cnt[r], je = (size_t)cnt[r + 1];
        while (i < ie || j < je) {
            int64_t ci = i < ie ? A->indices[i] : INT64_MAX, cj = j < je ? tix[j] : INT64_MAX;
            volatile double v;
            int64_t c;
            if (ci == cj) { v = A->data[i] + tdt[j]; c = ci; i++; j++; }
            else if (ci < cj) { v = A->data[i]; c = ci; i++; }
            else { v = tdt[j]; c = cj; j++; }
            v = v / 2.0;
            if (v != 0.0) { H->indices[out] = c; H->data[out] = v; out++; }
        }
    }
    H->indptr[n] = (int64_t)out; H->nnz = out;
    free(cnt); free(tix); free(tdt); free(pos);
    return H;
}

static void csr_hash(const csr *A, char hex[65]) {
    sha256_ctx c; sha256_init(&c);
    char sh[64]; int L = snprintf(sh, sizeof sh, "[%zu, %zu]", A->n, A->n);
    sha256_update(&c, sh, (size_t)L);
    nparr t; memset(&t, 0, sizeof t); t.ndim = 1; char hx[65];
    strcpy(t.descr, "<i8"); t.itemsize = 8; t.shape[0] = A->n + 1; t.count = A->n + 1; t.nbytes = 8 * (A->n + 1); t.data = (unsigned char *)A->indptr;
    array_hash(&t, hx); sha256_update(&c, hx, 64);
    t.shape[0] = A->nnz; t.count = A->nnz; t.nbytes = 8 * A->nnz; t.data = (unsigned char *)A->indices;
    array_hash(&t, hx); sha256_update(&c, hx, 64);
    strcpy(t.descr, "<f8"); t.data = (unsigned char *)A->data;
    array_hash(&t, hx); sha256_update(&c, hx, 64);
    sha256_final(&c, hex);
}

/* residual interval r = A x - mu x, entrywise, for column j of X (n x k) */
static void residual_interval(const csr *A, const nparr *X, size_t j, double mu, ival *r) {
    size_t n = A->n;
    double *x = xmalloc(n * sizeof *x);
    for (size_t i = 0; i < n; i++) x[i] = f64_2d(X, i, j);
    volatile double negmu = -mu;
    for (int mode = 0; mode < 2; mode++) {
        fesetround(mode ? FE_UPWARD : FE_DOWNWARD);
        for (size_t row = 0; row < n; row++) {
            volatile double acc = 0.0;
            for (size_t p = (size_t)A->indptr[row]; p < (size_t)A->indptr[row + 1]; p++) {
                volatile double t = A->data[p] * x[A->indices[p]];
                acc = acc + t;
            }
            volatile double t2 = negmu * x[row];
            acc = acc + t2;
            if (mode) r[row].hi = acc; else r[row].lo = acc;
        }
    }
    free(x);
}

/* 2-norm interval of a vector of entrywise intervals */
static ival norm_interval(const ival *r, size_t n) {
    ival out;
    fesetround(FE_UPWARD);
    { volatile double s = 0.0; for (size_t i = 0; i < n; i++) { ival a = iv_abs(r[i]); volatile double t = a.hi * a.hi; s = s + t; } out.hi = sqrt(s); }
    fesetround(FE_DOWNWARD);
    { volatile double s = 0.0; for (size_t i = 0; i < n; i++) { ival a = iv_abs(r[i]); volatile double t = a.lo * a.lo; s = s + t; } out.lo = sqrt(s); }
    return out;
}

static ival vec_norm_interval(const nparr *X, size_t j) {
    size_t n = X->shape[0]; ival out;
    fesetround(FE_UPWARD);
    { volatile double s = 0.0; for (size_t i = 0; i < n; i++) { volatile double v = f64_2d(X, i, j); volatile double t = v * v; s = s + t; } out.hi = sqrt(s); }
    fesetround(FE_DOWNWARD);
    { volatile double s = 0.0; for (size_t i = 0; i < n; i++) { volatile double v = f64_2d(X, i, j); volatile double t = v * v; s = s + t; } out.lo = sqrt(s); }
    return out;
}


/* Exact-sum machinery (error-free transformations, round-to-nearest): TwoSum (Knuth) and TwoProduct (Dekker
 * with Veltkamp splitting; -ffp-contract=off keeps the compiler from fusing). dot_interval returns an interval
 * containing the exact value of c + sum_i x_i y_i: the leading sum s is accumulated with TwoSum, every error
 * term is kept exactly, and the error terms are summed with directed rounding. */

static void two_sum(double a, double b, double *s, double *e) {
    volatile double x = a + b; volatile double bv = x - a; volatile double av = x - bv;
    volatile double e1 = a - av, e2 = b - bv; *s = x; *e = e1 + e2;
}
static void split(double a, double *hi, double *lo) {
    volatile double c = 134217729.0 * a; volatile double h = c - (c - a); *hi = h; *lo = a - h;
}
static void two_prod(double a, double b, double *p, double *e) {
    volatile double x = a * b; double ah, al, bh, bl; split(a, &ah, &al); split(b, &bh, &bl);
    volatile double t1 = ah * bh, t2 = ah * bl, t3 = al * bh, t4 = al * bl;
    volatile double r = ((t1 - x) + t2 + t3) + t4;
    *p = x; *e = r;
}
static ival dot_interval(const double *x, const double *y, size_t n, double c) {
    double *errs = xmalloc((2 * n + 2) * sizeof *errs); size_t ne = 0;
    fesetround(FE_TONEAREST);
    double s = c;
    for (size_t i = 0; i < n; i++) {
        double p, ep, es;
        two_prod(x[i], y[i], &p, &ep);
        two_sum(s, p, &s, &es);
        errs[ne++] = ep; errs[ne++] = es;
    }
    ival out;
    fesetround(FE_DOWNWARD);
    { volatile double a = 0.0; for (size_t i = 0; i < ne; i++) a = a + errs[i]; volatile double r = s + a; out.lo = r; }
    fesetround(FE_UPWARD);
    { volatile double a = 0.0; for (size_t i = 0; i < ne; i++) a = a + errs[i]; volatile double r = s + a; out.hi = r; }
    free(errs);
    return out;
}

/* Frobenius-norm interval of X_g^T X_g - I for the member columns idx[0..g) */
static ival gram_defect_interval(const nparr *X, const size_t *idx, size_t g) {
    size_t n = X->shape[0];
    ival *G = xmalloc(g * g * sizeof *G);
    double *xa = xmalloc(n * sizeof *xa), *xb = xmalloc(n * sizeof *xb);
    for (size_t a = 0; a < g; a++) {
        for (size_t i = 0; i < n; i++) xa[i] = f64_2d(X, i, idx[a]);
        for (size_t b = 0; b < g; b++) {
            for (size_t i = 0; i < n; i++) xb[i] = f64_2d(X, i, idx[b]);
            G[a * g + b] = dot_interval(xa, xb, n, a == b ? -1.0 : 0.0);
        }
    }
    ival out = norm_interval(G, g * g);
    free(G); free(xa); free(xb);
    return out;
}

/* Frobenius-norm interval of X_g^T R_g, R_g the residual intervals of the members (n x g) */
static ival xr_interval(const nparr *X, const size_t *idx, size_t g, ival **R) {
    size_t n = X->shape[0];
    ival *M = xmalloc(g * g * sizeof *M);
    for (size_t a = 0; a < g; a++)
        for (size_t b = 0; b < g; b++) {
            for (int mode = 0; mode < 2; mode++) {
                fesetround(mode ? FE_UPWARD : FE_DOWNWARD);
                volatile double s = 0.0;
                for (size_t i = 0; i < n; i++) {
                    double xv = f64_2d(X, i, idx[a]);
                    double rv = mode ? (xv >= 0 ? R[b][i].hi : R[b][i].lo) : (xv >= 0 ? R[b][i].lo : R[b][i].hi);
                    volatile double t = xv * rv; s = s + t;
                }
                if (mode) M[a * g + b].hi = s; else M[a * g + b].lo = s;
            }
        }
    ival out = norm_interval(M, g * g);
    free(M);
    return out;
}

/* ------------------------------------------------------------------ main */

static jv *find_statement(jv *stmts, const char *id) {
    for (size_t i = 0; i < jlen(stmts); i++) if (jstreq(jget(jidx(stmts, i), "id"), id)) return jidx(stmts, i);
    return NULL;
}

static jv *find_count(jv *counts, double sigma) {
    for (size_t i = 0; i < jlen(counts); i++) { jv *c = jidx(counts, i); if (jnum(jget(c, "sigma"), NAN) == sigma) return c; }
    return NULL;
}

static void json_escape(buf *b, const char *s) { canon_string(b, s, strlen(s)); }

static void usage(void) {
    fprintf(stderr, "usage: cev CERTIFICATE.json [--operator operator.npz] [--report out.txt] [--json out.json] [--dump-canonical body.json] [--quiet]\n");
    exit(2);
}

int main(int argc, char **argv) {
    const char *cert_path = NULL, *op_path = NULL, *report_path = NULL, *json_path = NULL, *dump_path = NULL; int quiet = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--operator") && i + 1 < argc) op_path = argv[++i];
        else if (!strcmp(argv[i], "--report") && i + 1 < argc) report_path = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--dump-canonical") && i + 1 < argc) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--version")) { printf("cev %s\n", CEV_VERSION); return 0; }
        else if (argv[i][0] == '-') usage();
        else if (!cert_path) cert_path = argv[i];
        else usage();
    }
    if (!cert_path) usage();
    if (fesetround(FE_UPWARD) != 0) die("fesetround is not supported on this platform");
    { volatile double one = 1.0, tiny = 1e-30; volatile double t = one + tiny; fesetround(FE_TONEAREST);
      if (!(t > 1.0)) die("directed rounding is not honoured by this build (compile with -frounding-math)"); }

    size_t clen; unsigned char *ctext = read_file(cert_path, &clen);
    jv *J = jparse((const char *)ctext, clen, cert_path);
    if (J->t != J_OBJ) die("certificate body is not a JSON object");
    if (!jstreq(jget(J, "schema"), "certify-ed/certificate/2.1"))
        add("INFO", "schema", "schema is %s (this verifier was written against certify-ed/certificate/2.1)", jstr(jget(J, "schema")) ? jstr(jget(J, "schema")) : "absent");

    /* ---- H1 body hash */
    {
        buf b = {0}; canon(&b, J, "integrity");
        char hex[65]; sha256_hex(b.p, b.n, hex);
        if (dump_path) { FILE *f = fopen(dump_path, "wb"); if (!f) die("cannot write %s", dump_path); fwrite(b.p, 1, b.n, f); fclose(f); }
        const char *rec = jstr(jget(jget(J, "integrity"), "body_sha256"));
        add(PASSFAIL(rec && !strcmp(rec, hex)), "H1.body_sha256", "recomputed %s, recorded %s", hex, rec ? rec : "(absent)");
        free(b.p);
    }

    /* ---- sidecar */
    jv *AB = jget(J, "arrays");
    nparr *arrs = NULL; int narr = 0;
    nparr *A_mu = NULL, *A_rad = NULL, *A_ex = NULL, *A_res = NULL, *A_X = NULL;
    if (jisnull(AB) || jisnull(jget(AB, "sidecar"))) {
        add("SKIP", "H2.arrays", "the certificate has no array sidecar");
    } else {
        char side[2048];
        const char *sc = jstr(jget(AB, "sidecar"));
        const char *slash = strrchr(cert_path, '/'); const char *bs = strrchr(cert_path, '\\');
        if (bs && (!slash || bs > slash)) slash = bs;
        if (slash) snprintf(side, sizeof side, "%.*s%s", (int)(slash - cert_path + 1), cert_path, sc);
        else snprintf(side, sizeof side, "%s", sc);
        load_npz(side, &arrs, &narr);
        jv *HS = jget(AB, "sha256"), *SH = jget(AB, "shapes"), *DT = jget(AB, "dtypes");
        for (size_t i = 0; i < jlen(HS); i++) {
            const char *name = HS->keys[i]; nparr *a = np_find(arrs, narr, name);
            char id[64]; snprintf(id, sizeof id, "H2.%s", name);
            if (!a) { add("FAIL", id, "array %s listed in the certificate is missing from %s", name, sc); continue; }
            char hex[65]; array_hash(a, hex);
            const char *rec = jstr(HS->items[i]);
            add(PASSFAIL(rec && !strcmp(rec, hex)), id, "recomputed %s, recorded %s", hex, rec ? rec : "(absent)");
            /* shape / dtype records */
            jv *shp = jget(SH, name); int ok = shp && (int)jlen(shp) == a->ndim;
            for (int d = 0; ok && d < a->ndim; d++) ok = jnum(jidx(shp, (size_t)d), -1) == (double)a->shape[d];
            ok = ok && jstreq(jget(DT, name), a->descr);
            snprintf(id, sizeof id, "C2.%s", name);
            add(PASSFAIL(ok), id, "shape/dtype record vs member: %s %dd (%s)", a->descr, a->ndim, ok ? "match" : "mismatch");
        }
        for (int i = 0; i < narr; i++) if (!jget(HS, arrs[i].name)) add("FAIL", "H2.extra", "sidecar member %s is not listed in the certificate", arrs[i].name);
        A_mu = np_find(arrs, narr, "eigenvalues"); A_rad = np_find(arrs, narr, "radius"); A_ex = np_find(arrs, narr, "existence_radius");
        A_res = np_find(arrs, narr, "residual_bound"); A_X = np_find(arrs, narr, "eigenvectors");
        if (A_mu && strcmp(A_mu->descr, "<f8")) die("eigenvalues array dtype %s is not supported (float64 only)", A_mu->descr);
        if (A_X && strcmp(A_X->descr, "<f8")) { add("SKIP", "R.vectors", "stored eigenvectors have dtype %s; only real float64 vectors are recomputed", A_X->descr); A_X = NULL; }
    }

    jv *CL = jget(J, "claim"), *SUBJ = jget(J, "subject"), *ENC = jget(J, "enclosures"), *STM = jget(J, "statements");
    jv *S2 = find_statement(STM, "S2.eigenpairs"), *S3 = find_statement(STM, "S3.completeness");
    jv *EV2 = S2 ? jget(S2, "evidence") : NULL, *EV3 = S3 ? jget(S3, "evidence") : NULL;
    const char *st2 = jstr(jget(S2, "status")), *st3 = jstr(jget(S3, "status"));
    jv *CLmu = jget(CL, "eigenvalues");
    size_t k = jlen(CLmu);
    double *mu = xmalloc((k + 1) * sizeof *mu);
    for (size_t i = 0; i < k; i++) mu[i] = jnum(jidx(CLmu, i), NAN);
    int sector = !jisnull(jget(SUBJ, "sector"));
    double model_shift = jnum(jget(EV2, "model_shift"), 0.0);
    double eps_s3 = sector ? 0.0 : model_shift;

    /* ---- C1 claim vs arrays */
    if (A_mu) {
        int ok = A_mu->count == k;
        for (size_t i = 0; ok && i < k; i++) ok = f64_at(A_mu, i) == mu[i];
        add(PASSFAIL(ok), "C1.claim_eigenvalues", "claim.eigenvalues (%zu) %s arrays.eigenvalues (%zu), bitwise", k, ok ? "==" : "!=", A_mu->count);
        add(PASSFAIL(jnum(jget(CL, "n_eigenvalues"), -1) == (double)k), "C1.n_eigenvalues", "claim.n_eigenvalues = %g, k = %zu", jnum(jget(CL, "n_eigenvalues"), -1), k);
    }
    /* ---- H3 eigenvectors hash */
    if (A_X) {
        char hex[65]; array_hash(A_X, hex);
        const char *rec = jstr(jget(CL, "eigenvectors_sha256"));
        add(PASSFAIL(rec && !strcmp(rec, hex)), "H3.eigenvectors_sha256", "stored vectors hash %s, claim records %s", hex, rec ? rec : "(absent)");
        if (A_X->ndim != 2 || A_X->shape[1] != k) add("FAIL", "C2.eigenvectors_shape", "stored eigenvectors are %dd with %zu columns, k = %zu", A_X->ndim, A_X->ndim == 2 ? A_X->shape[1] : 0, k);
    }
    /* ---- H4 spec hash */
    if (!jisnull(jget(SUBJ, "spec"))) {
        buf b = {0}; canon(&b, jget(SUBJ, "spec"), NULL);
        char hex[65]; sha256_hex(b.p, b.n, hex);
        const char *rec = jstr(jget(SUBJ, "spec_sha256"));
        add(PASSFAIL(rec && !strcmp(rec, hex)), "H4.spec_sha256", "canonical spec hash %s, recorded %s", hex, rec ? rec : "(absent)");
        free(b.p);
    } else add("SKIP", "H4.spec_sha256", "no model specification in the certificate");

    /* ---- enclosures: per-pair records, arrays, groups */
    size_t nenc = jlen(ENC);
    jv **encof = xmalloc((k + 1) * sizeof *encof); memset(encof, 0, (k + 1) * sizeof *encof);
    int have_enc = nenc > 0 && A_rad && A_ex && A_res && A_mu;
    if (nenc == 0) add("SKIP", "C3.enclosures", "no enclosures in the certificate (S2 %s)", st2 ? st2 : "absent");
    else if (!have_enc) add("FAIL", "C3.enclosures", "enclosures present but the radius/existence_radius/residual_bound arrays are not all stored");
    else {
        int ok_idx = nenc == k, ok_rec = 1, ok_arr = 1, ok_mono = 1; char why[300] = "";
        for (size_t e = 0; e < nenc && ok_idx; e++) {
            jv *r = jidx(ENC, e); double di = jnum(jget(r, "index"), -1);
            if (di < 0 || di >= (double)k || encof[(size_t)di]) { ok_idx = 0; break; }
            encof[(size_t)di] = r;
        }
        if (!ok_idx) add("FAIL", "C3.index", "enclosures do not index the %zu claimed pairs exactly once (%zu records)", k, nenc);
        else {
            for (size_t i = 0; i < k; i++) {
                jv *r = encof[i];
                double rad = jnum(jget(r, "radius"), NAN), ex = jnum(jget(r, "existence_radius"), NAN), rb = jnum(jget(r, "residual_bound"), NAN);
                double lo = jnum(jget(r, "lower"), NAN), hi = jnum(jget(r, "upper"), NAN), cl = jnum(jget(r, "claimed"), NAN);
                if (cl != mu[i]) { ok_rec = 0; snprintf(why, sizeof why, "pair %zu: claimed %.17g != eigenvalue %.17g", i, cl, mu[i]); }
                if (!(lo <= rd_sub(mu[i], rad)) || !(hi >= ru_add(mu[i], rad))) { ok_rec = 0; snprintf(why, sizeof why, "pair %zu: [lower, upper] does not contain mu -+ radius", i); }
                if (!(ex <= rad)) { ok_mono = 0; snprintf(why, sizeof why, "pair %zu: existence_radius %.17g > radius %.17g", i, ex, rad); }
                /* arrays: radius and existence radius carry the model shift (add_up); residual_bound is shift-free */
                if (!(f64_at(A_rad, i) >= ru_add(rad, eps_s3 == 0.0 && sector ? jnum(jget(EV2, "sector_lift_shift"), 0.0) : model_shift))) { ok_arr = 0; snprintf(why, sizeof why, "pair %zu: arrays.radius %.17g < enclosure radius %.17g + shift %.3g", i, f64_at(A_rad, i), rad, model_shift); }
                if (!(f64_at(A_ex, i) >= ru_add(ex, eps_s3 == 0.0 && sector ? jnum(jget(EV2, "sector_lift_shift"), 0.0) : model_shift))) { ok_arr = 0; snprintf(why, sizeof why, "pair %zu: arrays.existence_radius below enclosure existence radius + shift", i); }
                if (f64_at(A_res, i) != rb) { ok_arr = 0; snprintf(why, sizeof why, "pair %zu: arrays.residual_bound %.17g != enclosure residual_bound %.17g", i, f64_at(A_res, i), rb); }
                if (!(f64_at(A_ex, i) <= f64_at(A_rad, i))) { ok_mono = 0; snprintf(why, sizeof why, "pair %zu: arrays.existence_radius > arrays.radius", i); }
            }
            if (ok_rec) add("PASS", "C3.enclosure_records", "%zu records: claimed == eigenvalue, lower <= RD(mu - r), upper >= RU(mu + r)", k);
            else add("FAIL", "C3.enclosure_records", "%s", why);
            if (ok_mono) add("PASS", "C3.radius_order", "existence_radius <= radius for every pair (records and arrays)");
            else add("FAIL", "C3.radius_order", "%s", why);
            if (ok_arr) add("PASS", "C4.arrays_vs_records", "arrays.radius/existence_radius >= RU(record + model shift %.3g); residual_bound equal", model_shift);
            else add("FAIL", "C4.arrays_vs_records", "%s", why);
        }
    }

    /* groups (S2 evidence) and the recorded cluster arithmetic */
    jv *GR = jget(EV2, "groups");
    double Hn = jnum(jget(jget(SUBJ, "operator"), "norm_upper_bound"), NAN);
    size_t ngroups = jlen(GR);
    for (size_t gi = 0; gi < ngroups && have_enc && encof[0]; gi++) {
        jv *g = jidx(GR, gi); jv *mem = jget(g, "members"); size_t gsz = jlen(mem);
        char id[64]; snprintf(id, sizeof id, "G%zu", gi);
        double grad = jnum(jget(g, "radius"), NAN), eta = jnum(jget(g, "eta"), NAN);
        const char *method = jstr(jget(g, "method")); int distinct = jbool(jget(g, "distinct"), 0);
        int ok = gsz >= 2; double mumax = -INFINITY, mumin = INFINITY, lam = 0.0;
        for (size_t m = 0; m < gsz; m++) {
            double di = jnum(jidx(mem, m), -1);
            if (di < 0 || di >= (double)k) { ok = 0; break; }
            size_t i = (size_t)di; jv *r = encof[i];
            if (jnum(jget(r, "radius"), NAN) != grad && !strcmp(method ? method : "", "cluster")) ok = 0;
            if (jbool(jget(r, "distinct"), 0) != distinct) ok = 0;
            if (!jstreq(jget(r, "method"), method ? method : "")) ok = 0;
            if (jnum(jget(r, "group_size"), -1) != (double)gsz) ok = 0;
            if (mu[i] > mumax) mumax = mu[i];
            if (mu[i] < mumin) mumin = mu[i];
            if (fabs(mu[i]) > lam) lam = fabs(mu[i]);
        }
        char cid[80]; snprintf(cid, sizeof cid, "C5.%s.members", id);
        add(PASSFAIL(ok), cid, "group of %zu (%s, distinct=%d): member records agree with the group", gsz, method ? method : "?", distinct);
        if (!ok) continue;
        if (!strcmp(method ? method : "", "cluster")) {
            jv *bd = jget(g, "bound");
            double R_fro = jnum(jget(bd, "R_fro"), NAN), XR = jnum(jget(bd, "XR"), NAN), delta = jnum(jget(bd, "delta"), NAN);
            double spread = jnum(jget(bd, "spread"), NAN), rho = jnum(jget(bd, "rho"), NAN), eps_star = jnum(jget(bd, "eps_star"), NAN);
            /* R_fro >= || (residual_bound_m) ||_2 */
            fesetround(FE_UPWARD);
            volatile double s = 0.0;
            for (size_t m = 0; m < gsz; m++) { volatile double v = jnum(jget(encof[(size_t)jnum(jidx(mem, m), 0)], "residual_bound"), NAN); volatile double t = v * v; s = s + t; }
            double rf_up = sqrt(s);
            snprintf(cid, sizeof cid, "G%zu.R_fro", gi); add(PASSFAIL(R_fro >= rf_up), cid, "recorded R_fro %.6e >= RU ||residual bounds||_2 %.6e", R_fro, rf_up);
            /* eta < 1 and delta >= 1 - sqrt(1 - eta) */
            double one_m = rd_sub(1.0, eta), sq_dn = rd_sqrt(one_m), d_up = ru_sub(1.0, sq_dn);
            snprintf(cid, sizeof cid, "G%zu.eta", gi); add(PASSFAIL(eta >= 0 && eta < 1.0), cid, "recorded eta %.6e < 1", eta);
            snprintf(cid, sizeof cid, "G%zu.delta", gi); add(PASSFAIL(delta >= d_up), cid, "recorded delta %.6e >= RU(1 - sqrt(1 - eta)) %.6e", delta, d_up);
            /* rho >= R_fro + (||H|| + ||Lambda||) delta */
            double rho_v = ru_add(R_fro, ru_mul(ru_add(Hn, lam), delta));
            snprintf(cid, sizeof cid, "G%zu.rho", gi); add(PASSFAIL(rho >= rho_v && !isnan(Hn)), cid, "recorded rho %.6e >= RU(R_fro + (%.6g + %.6g) delta) %.6e", rho, Hn, lam, rho_v);
            /* spread >= max mu - min mu */
            double sp_v = ru_sub(mumax, mumin);
            snprintf(cid, sizeof cid, "G%zu.spread", gi); add(PASSFAIL(spread >= sp_v), cid, "recorded spread %.6e >= RU(max - min) %.6e", spread, sp_v);
            /* eps_star >= R_fro / sqrt(1-eta) + XR / (1-eta) + delta spread / sqrt(1-eta) */
            double inv_sqrt = ru_div(1.0, sq_dn), inv = ru_div(1.0, one_m);
            double es_v = ru_add(ru_add(ru_mul(R_fro, inv_sqrt), ru_mul(XR, inv)), ru_mul(ru_mul(delta, spread), inv_sqrt));
            snprintf(cid, sizeof cid, "G%zu.eps_star", gi); add(PASSFAIL(eps_star >= es_v), cid, "recorded eps_star %.6e >= RU(formula) %.6e", eps_star, es_v);
            double mn = rho < eps_star ? rho : eps_star;
            snprintf(cid, sizeof cid, "G%zu.radius", gi); add(PASSFAIL(grad >= mn), cid, "group radius %.6e >= min(rho, eps_star) %.6e", grad, mn);
        } else {
            snprintf(cid, sizeof cid, "G%zu.existence_only", gi);
            add(PASSFAIL(!distinct), cid, "existence-only group (eta >= 1 or no norm bound) is not marked distinct");
        }
    }

    /* C6 disjointness of the union intervals of groups marked distinct (pre-shift records, as certify-ed) */
    if (have_enc && encof[0]) {
        /* group id per pair */
        size_t ng = 0; double *L = xmalloc((k + 1) * sizeof *L), *Uu = xmalloc((k + 1) * sizeof *Uu); int *dist = xmalloc((k + 1) * sizeof *dist);
        int *gid = xmalloc((k + 1) * sizeof *gid);
        for (size_t i = 0; i < k; i++) { gid[i] = (int)jnum(jget(encof[i], "group"), -1); if ((size_t)gid[i] + 1 > ng) ng = (size_t)gid[i] + 1; }
        for (size_t g = 0; g < ng; g++) { L[g] = INFINITY; Uu[g] = -INFINITY; dist[g] = 1; }
        for (size_t i = 0; i < k; i++) {
            size_t g = (size_t)gid[i];
            double lo = jnum(jget(encof[i], "lower"), NAN), hi = jnum(jget(encof[i], "upper"), NAN);
            if (lo < L[g]) L[g] = lo;
            if (hi > Uu[g]) Uu[g] = hi;
            if (!jbool(jget(encof[i], "distinct"), 0)) dist[g] = 0;
        }
        int ok = 1; char why[200] = "";
        for (size_t a = 0; a < ng && ok; a++)
            for (size_t b = 0; b < ng && ok; b++)
                if (a != b && dist[a] && !(Uu[a] < L[b] || Uu[b] < L[a])) { ok = 0; snprintf(why, sizeof why, "group %zu marked distinct overlaps group %zu", a, b); }
        int all_fin = 1;
        for (size_t i = 0; i < k; i++) if (jbool(jget(encof[i], "distinct"), 0) && !isfinite(jnum(jget(encof[i], "radius"), NAN))) all_fin = 0;
        if (ok && all_fin) add("PASS", "C6.disjoint", "%zu groups; every group marked distinct has a finite radius and is disjoint from every other group", ng);
        else add("FAIL", "C6.disjoint", "%s", ok ? "a group marked distinct has a non-finite radius" : why);
        free(L); free(Uu); free(dist); free(gid);
    }

    /* ---- operator and recomputation of the Theorem-1 radii */
    csr *A = NULL; int herm = !jisnull(jget(J, "hermitian_part"));
    if (op_path) {
        const char *why = NULL; csr *A0 = csr_load(op_path, &why);
        if (!A0) add("SKIP", "H5.operator_sha256", "operator file not usable: %s", why);
        else {
            if (herm) { csr *Hp = csr_hermitian_part(A0); A = Hp; } else A = A0;
            char hex[65]; csr_hash(A, hex);
            const char *rec = jstr(jget(jget(SUBJ, "operator"), "matrix_sha256"));
            int ok = rec && !strcmp(rec, hex);
            add(PASSFAIL(ok), "H5.operator_sha256", "%s of the operator file: %s, certificate records %s", herm ? "Hermitian part" : "canonical CSR", hex, rec ? rec : "(absent)");
            add(PASSFAIL(jnum(jget(jget(SUBJ, "operator"), "dimension"), -1) == (double)A->n), "C2.dimension", "operator dimension %zu, certificate records %g", A->n, jnum(jget(jget(SUBJ, "operator"), "dimension"), -1));
            if (!ok) A = NULL;    /* a matrix that is not the certified one proves nothing about the radii */
        }
    }
    ival **Rcols = NULL;
    if (A && A_X && have_enc && encof[0]) {
        if (A_X->shape[0] != A->n) add("FAIL", "C2.vector_length", "stored vectors have %zu rows, operator dimension %zu", A_X->shape[0], A->n);
        else {
            Rcols = xmalloc(k * sizeof *Rcols);
            int ok_res = 1, ok_ex = 1; char why[400] = ""; double slack_res = INFINITY, slack_ex = INFINITY; int proven_unsound = 0;
            for (size_t i = 0; i < k; i++) {
                Rcols[i] = xmalloc(A->n * sizeof **Rcols);
                residual_interval(A, A_X, i, mu[i], Rcols[i]);
                ival rn = norm_interval(Rcols[i], A->n), xn = vec_norm_interval(A_X, i);
                double rb = jnum(jget(encof[i], "residual_bound"), NAN), ex = jnum(jget(encof[i], "existence_radius"), NAN);
                double rad_up = ru_div(rn.hi, xn.lo), rad_dn = rd_div(rn.lo, xn.hi);
                if (!(rb >= rn.hi)) { ok_res = 0; if (rn.lo > rb) proven_unsound = 1; snprintf(why, sizeof why, "pair %zu: recorded residual_bound %.6e, recomputed ||Ax - mu x|| in [%.6e, %.6e]", i, rb, rn.lo, rn.hi); }
                if (!(ex >= rad_up)) { ok_ex = 0; if (rad_dn > ex) proven_unsound = 1; snprintf(why + strlen(why), sizeof why - strlen(why), "%spair %zu: recorded existence_radius %.6e, recomputed Theorem-1 radius in [%.6e, %.6e]", why[0] ? "; " : "", i, ex, rad_dn, rad_up); }
                if (rb - rn.hi < slack_res) slack_res = rb - rn.hi;
                if (ex - rad_up < slack_ex) slack_ex = ex - rad_up;
            }
            if (ok_res) add("PASS", "R2.residual_bound", "%zu pairs: recorded residual_bound >= RU ||A x - mu x||_2 (smallest slack %.3e)", k, slack_res);
            else add("FAIL", "R2.residual_bound", "%s%s", why, proven_unsound ? " [a recorded bound lies below the directed-rounding LOWER bound: unsound]" : " [not confirmed: a recorded bound lies inside the rounding interval]");
            if (ok_ex) add("PASS", "R3.existence_radius", "%zu pairs: recorded existence_radius >= RU(||r|| / ||x||) (smallest slack %.3e)", k, slack_ex);
            else add("FAIL", "R3.existence_radius", "%s%s", why, proven_unsound ? " [unsound]" : " [not confirmed]");
            /* R4/R5: Gram defect and X^H R per cluster group */
            for (size_t gi = 0; gi < ngroups; gi++) {
                jv *g = jidx(GR, gi); jv *mem = jget(g, "members"); size_t gsz = jlen(mem);
                if (!jstreq(jget(g, "method"), "cluster") || gsz < 2) continue;
                size_t *idx = xmalloc(gsz * sizeof *idx); ival **Rg = xmalloc(gsz * sizeof *Rg);
                for (size_t m = 0; m < gsz; m++) { idx[m] = (size_t)jnum(jidx(mem, m), 0); Rg[m] = Rcols[idx[m]]; }
                double eta = jnum(jget(g, "eta"), NAN), XR = jnum(jget(jget(g, "bound"), "XR"), NAN);
                ival gd = gram_defect_interval(A_X, idx, gsz), xr = xr_interval(A_X, idx, gsz, Rg);
                char cid[64];
                snprintf(cid, sizeof cid, "R4.G%zu.eta", gi);
                add(PASSFAIL(eta >= gd.hi), cid, "recorded eta %.6e vs recomputed ||X^T X - I||_F in [%.6e, %.6e]%s", eta, gd.lo, gd.hi, eta >= gd.hi ? "" : (gd.lo > eta ? " [unsound]" : " [not confirmed]"));
                snprintf(cid, sizeof cid, "R5.G%zu.XR", gi);
                add(PASSFAIL(XR >= xr.hi), cid, "recorded XR %.6e vs recomputed ||X^T R||_F in [%.6e, %.6e]%s", XR, xr.lo, xr.hi, XR >= xr.hi ? "" : (xr.lo > XR ? " [unsound]" : " [not confirmed]"));
                free(idx); free(Rg);
            }
        }
    } else if (op_path && !A_X) add("SKIP", "R2.residual_bound", "eigenvectors are not stored in the sidecar: Theorem-1 radii not recomputed (hash chain and recorded arithmetic checked instead)");
    else if (!op_path) add("SKIP", "R2.residual_bound", "no --operator given: Theorem-1 radii not recomputed (hash chain and recorded arithmetic checked instead)");

    /* ---- U1 no-underflow hypothesis (INT-01): the standard-model bounds assume every nonzero operand magnitude is
     * >= the recorded threshold (2^-900 by default); the stored eigenvalues and vectors are scanned here */
    jv *AR = jget(J, "arithmetic"); jv *UC = jget(AR, "underflow_check");
    double thr = jnum(jget(UC, "threshold"), ldexp(1.0, -900));
    int tiny_found = 0; size_t tiny_count = 0; double tiny_min = INFINITY;
    for (size_t i = 0; A_mu && i < A_mu->count; i++) { double v = fabs(f64_at(A_mu, i)); if (v != 0 && v < thr) { tiny_found = 1; tiny_count++; if (v < tiny_min) tiny_min = v; } }
    for (size_t i = 0; A_X && i < A_X->count; i++) { double v = fabs(f64_at(A_X, i)); if (v != 0 && v < thr) { tiny_found = 1; tiny_count++; if (v < tiny_min) tiny_min = v; } }
    int recorded_findings = jlen(jget(UC, "findings")) > 0 || jlen(jget(jget(EV2, "underflow"), "findings")) > 0;
    if (A_X && tiny_found) add("INFO", "U1.underflow_scan", "%zu nonzero magnitudes below the threshold %.3e in the stored eigenvalues/vectors (min %.3e); recorded findings: %d", tiny_count, thr, tiny_min, recorded_findings);
    else if (A_X) add("PASS", "U1.underflow_scan", "no nonzero magnitude below %.3e in the stored eigenvalues and vectors; recorded findings: %d", thr, recorded_findings);
    else add("SKIP", "U1.underflow_scan", "eigenvectors not stored: the no-underflow hypothesis is taken from the record (findings: %d)", recorded_findings);
    int hypotheses_ok = !recorded_findings && !tiny_found && jbool(jget(jget(AR, "runtime_probe"), "standard_model_hypotheses_hold"), 1);

    /* ---- S2 status consistency */
    double target = jnum(jget(jget(EV2, "accuracy_target"), "value"), NAN);
    if (st2 && !strcmp(st2, "CERTIFIED")) {
        if (!have_enc || !encof[0]) add("FAIL", "S2.status", "S2 CERTIFIED without complete enclosure records");
        else {
            int all_dist = 1, all_fin = 1; double rmax = 0;
            for (size_t i = 0; i < k; i++) {
                if (!jbool(jget(encof[i], "distinct"), 0)) all_dist = 0;
                double r = f64_at(A_rad, i); if (!isfinite(r)) all_fin = 0; if (r > rmax) rmax = r;
            }
            int ok = all_dist && all_fin && isfinite(target) && rmax <= target && hypotheses_ok;
            add(PASSFAIL(ok), "S2.status", "CERTIFIED permitted: all distinct %d, all finite %d, max arrays.radius %.6e <= accuracy target %.3e: %d, floating-point hypotheses (no underflow finding, probe) hold: %d", all_dist, all_fin, rmax, target, rmax <= target, hypotheses_ok);
            add(PASSFAIL(jbool(jget(EV2, "rigorous"), 0)), "S2.rigorous", "evidence.rigorous is %s", jbool(jget(EV2, "rigorous"), 0) ? "true" : "false");
        }
    } else if (st2 && !strcmp(st2, "FAILED")) {
        if (have_enc && encof[0]) {
            double rmax = 0; size_t worst = 0;
            for (size_t i = 0; i < k; i++) { double r = f64_at(A_rad, i); if (r > rmax) { rmax = r; worst = i; } }
            int rep = isfinite(target) && rmax > target;
            add(rep ? "REPRODUCED" : "NOT_REPRODUCED", "S2.failed", "max arrays.radius %.6e (pair %zu) %s accuracy target %.3e", rmax, worst, rep ? ">" : "<=", target);
            if (Rcols) {
                /* independent evidence: the largest directed-rounding LOWER bound of a Theorem-1 radius over all pairs */
                double best = -1; size_t bi = 0;
                for (size_t i = 0; i < k; i++) {
                    ival rn = norm_interval(Rcols[i], A->n), xn = vec_norm_interval(A_X, i);
                    double rad_dn = rd_div(rn.lo, xn.hi);
                    if (rad_dn > best) { best = rad_dn; bi = i; }
                }
                add("INFO", "S2.failed.independent", "pair %zu: directed-rounding LOWER bound of its Theorem-1 radius %.6e %s the target %.3e (%s)", bi, best, best > target ? ">" : "<=", target, best > target ? "a claimed value is wrong independently of certify-ed's bounds" : "no single Theorem-1 lower bound exceeds the target; the failure rests on the cluster bound or on certify-ed's rounding terms");
            }
        } else add("INFO", "S2.failed", "S2 FAILED; no enclosure arrays to reproduce it from");
    } else add("INFO", "S2.status", "S2 is %s: no certification to re-check", st2 ? st2 : "absent");

    /* ---- S3 status consistency */
    const char *kind = jstr(jget(CL, "kind"));
    double *hiv = xmalloc((k + 1) * sizeof *hiv), *lov = xmalloc((k + 1) * sizeof *lov);
    int radii_ok = have_enc && encof[0];
    for (size_t i = 0; radii_ok && i < k; i++) {
        double rad = sector ? jnum(jget(encof[i], "radius"), NAN) : f64_at(A_rad, i);
        hiv[i] = ru_add(mu[i], rad); lov[i] = rd_sub(mu[i], rad);
    }
    if (sector && radii_ok) add("INFO", "S3.sector", "sector certificate: S3 re-checked with the pre-lift radii of the sector block, as certify-ed does");
    jv *counts = jget(EV3, "counts");
    if (st3 && !strcmp(st3, "CERTIFIED") && kind && !strcmp(kind, "lowest")) {
        if (!radii_ok) add("FAIL", "S3.status", "S3 CERTIFIED (lowest) without enclosures");
        else {
            double sigma = jnum(jget(EV3, "top_shift"), NAN), beta = jnum(jget(EV3, "beta_effective"), NAN), nelb = jnum(jget(EV3, "next_eigenvalue_lower_bound"), NAN);
            jv *c = find_count(counts, sigma);
            int c_ok = c && jbool(jget(c, "conclusive"), 0) && !jbool(jget(c, "resource_failure"), 0) && jnum(jget(c, "negative"), -1) == (double)k;
            add(PASSFAIL(c_ok), "S3.inertia", "count at sigma = %.17g: negative = %g, conclusive = %d; k = %zu", sigma, c ? jnum(jget(c, "negative"), -1) : -1.0, c ? jbool(jget(c, "conclusive"), 0) : 0, k);
            double beta_v = c ? ru_add(jnum(jget(c, "beta"), NAN), eps_s3) : NAN;
            add(PASSFAIL(beta >= beta_v), "S3.beta", "beta_effective %.6e >= RU(count beta + shift %.3g) %.6e", beta, eps_s3, beta_v);
            add(PASSFAIL(nelb <= rd_sub(sigma, beta)), "S3.bound", "next_eigenvalue_lower_bound %.17g <= RD(sigma - beta) %.17g", nelb, rd_sub(sigma, beta));
            double hik = -INFINITY; int all_dist = 1;
            for (size_t i = 0; i < k; i++) { if (hiv[i] > hik) hik = hiv[i]; if (!jbool(jget(encof[i], "distinct"), 0)) all_dist = 0; }
            add(PASSFAIL(hik < nelb && all_dist), "S3.ordering", "max RU(mu + radius) %.17g < sigma - beta bound %.17g: %d; all enclosures distinct: %d", hik, nelb, hik < nelb, all_dist);
            double gap = jnum(jget(EV3, "gap_lower_bound"), NAN);
            add(PASSFAIL(gap <= rd_sub(nelb, hik)), "S3.gap", "gap_lower_bound %.6e <= RD(bound - max upper end) %.6e", gap, rd_sub(nelb, hik));
            jv *cw = jget(EV3, "certified_window");
            add(PASSFAIL(jstreq(jidx(cw, 0), "-inf") && jnum(jidx(cw, 1), NAN) == nelb), "S3.window_record", "certified_window is [-inf, next_eigenvalue_lower_bound]");
        }
    } else if (st3 && !strcmp(st3, "CERTIFIED") && kind && !strcmp(kind, "window")) {
        jv *W = jget(EV3, "window"); double a = jnum(jidx(W, 0), NAN), b = jnum(jidx(W, 1), NAN);
        if (!radii_ok) add(jisnull(jget(EV3, "count_in_window")) ? "FAIL" : "SKIP", "S3.status", "S3 CERTIFIED (window) %s", jisnull(jget(EV3, "count_in_window")) ? "without enclosures" : "is a vectorless window count (C-02): edge logic not re-checked by this verifier");
        else {
            jv *ca = find_count(counts, a), *cb = find_count(counts, b);
            int c_ok = ca && cb && jbool(jget(ca, "conclusive"), 0) && jbool(jget(cb, "conclusive"), 0);
            double n_in = c_ok ? jnum(jget(cb, "negative"), 0) - jnum(jget(ca, "negative"), 0) : NAN;
            add(PASSFAIL(c_ok && n_in == jnum(jget(EV3, "count_from_inertia"), NAN)), "S3.inertia", "counts at a = %.10g and b = %.10g conclusive: %d; nu_b - nu_a = %g, recorded %g", a, b, c_ok, n_in, jnum(jget(EV3, "count_from_inertia"), NAN));
            double ba = jnum(jget(EV3, "beta_a"), NAN), bb = jnum(jget(EV3, "beta_b"), NAN);
            int b_ok = c_ok && ba >= ru_add(jnum(jget(ca, "beta"), NAN), eps_s3) && bb >= ru_add(jnum(jget(cb, "beta"), NAN), eps_s3);
            add(PASSFAIL(b_ok), "S3.beta", "beta_a %.3e, beta_b %.3e >= RU(count beta + shift %.3g)", ba, bb, eps_s3);
            double lo_w = ru_add(a, ba), hi_w = rd_sub(b, bb); int cin = 0, all_dist = 1;
            for (size_t i = 0; i < k; i++) { if (lov[i] > lo_w && hiv[i] < hi_w) cin++; if (!jbool(jget(encof[i], "distinct"), 0)) all_dist = 0; }
            add(PASSFAIL(all_dist && (double)cin == n_in), "S3.ordering", "%d enclosures strictly inside (RU(a + beta_a), RD(b - beta_b)) = (%.10g, %.10g), inertia count %g, all distinct %d", cin, lo_w, hi_w, n_in, all_dist);
            jv *cw = jget(EV3, "certified_window");
            add(PASSFAIL(jnum(jidx(cw, 0), -INFINITY) >= lo_w && jnum(jidx(cw, 1), INFINITY) <= hi_w), "S3.window_record", "certified_window [%.10g, %.10g] inside [RU(a + beta_a), RD(b - beta_b)]", jnum(jidx(cw, 0), NAN), jnum(jidx(cw, 1), NAN));
        }
    } else if (st3 && !strcmp(st3, "CERTIFIED") && kind && !strcmp(kind, "all")) {
        int all_dist = radii_ok; double dim = jnum(jget(jget(SUBJ, "operator"), "dimension"), -1);
        for (size_t i = 0; radii_ok && i < k; i++) if (!jbool(jget(encof[i], "distinct"), 0)) all_dist = 0;
        add(PASSFAIL(all_dist && dim == (double)k), "S3.status", "full-spectrum claim: all %zu enclosures distinct %d, dimension %g", k, all_dist, dim);
    } else if (st3 && !strcmp(st3, "CERTIFIED")) {
        add("SKIP", "S3.status", "S3 CERTIFIED for claim kind %s: not re-checked by this verifier", kind ? kind : "?");
    } else if (st3 && !strcmp(st3, "FAILED")) {
        if (!radii_ok) add("INFO", "S3.failed", "S3 FAILED; no enclosures to reproduce it from");
        else if (!jisnull(jget(EV3, "missing_below_top_claim_at_least"))) {
            double s_low = jnum(jget(EV3, "witness_shift"), NAN); jv *c = find_count(counts, s_low);
            double m_rec = jnum(jget(EV3, "missing_below_top_claim_at_least"), NAN);
            if (!c || !jbool(jget(c, "conclusive"), 0)) add("NOT_REPRODUCED", "S3.failed", "no conclusive count at the witness shift %.17g", s_low);
            else {
                double bl = ru_add(jnum(jget(c, "beta"), NAN), eps_s3), lo_top = NAN; size_t top = 0;
                for (size_t i = 0; i < k; i++) if (i == 0 || mu[i] > mu[top]) top = i;
                lo_top = lov[top];
                int collar_ok = bl < rd_sub(lo_top, s_low);
                int cnt = 0; for (size_t i = 0; i < k; i++) if (lov[i] <= ru_add(s_low, bl)) cnt++;
                double miss = jnum(jget(c, "negative"), 0) - cnt;
                int rep = collar_ok && miss >= 1;
                add(rep ? "REPRODUCED" : "NOT_REPRODUCED", "S3.failed", "inertia %g below witness shift %.10g (beta %.3e) vs %d claimed enclosures reaching below it: at least %g unclaimed eigenvalue(s) below the top claim (recorded %g); collar below the top enclosure: %d", jnum(jget(c, "negative"), 0), s_low, bl, cnt, miss, m_rec, collar_ok);
            }
        } else if (!jisnull(jget(EV3, "missing_at_least")) && kind && !strcmp(kind, "window")) {
            jv *W = jget(EV3, "window"); double a = jnum(jidx(W, 0), NAN), b = jnum(jidx(W, 1), NAN);
            jv *ca = find_count(counts, a), *cb = find_count(counts, b);
            double ba = jnum(jget(EV3, "beta_a"), NAN), bb = jnum(jget(EV3, "beta_b"), NAN);
            int c_ok = ca && cb && jbool(jget(ca, "conclusive"), 0) && jbool(jget(cb, "conclusive"), 0);
            double n_in = c_ok ? jnum(jget(cb, "negative"), 0) - jnum(jget(ca, "negative"), 0) : NAN;
            int touching = 0;
            for (size_t i = 0; i < k; i++) if (hiv[i] >= rd_sub(a, ba) && lov[i] <= ru_add(b, bb)) touching++;
            int rep = c_ok && n_in - touching >= 1;
            add(rep ? "REPRODUCED" : "NOT_REPRODUCED", "S3.failed", "inertia count in the window %g vs %d claimed enclosures touching it: at least %g unclaimed (recorded %g)", n_in, touching, n_in - touching, jnum(jget(EV3, "missing_at_least"), NAN));
        } else add("NOT_REPRODUCED", "S3.failed", "S3 FAILED with an evidence form this verifier does not re-derive (%s)", kind ? kind : "?");
    } else add("INFO", "S3.status", "S3 is %s: no certification to re-check", st3 ? st3 : "absent");
    if (st3 && !strcmp(st3, "CERTIFIED") && kind && strcmp(kind, "subset"))
        add(PASSFAIL(hypotheses_ok), "S3.hypotheses", "S3 CERTIFIED requires the floating-point hypotheses of S2 (no underflow finding, runtime probe): %d", hypotheses_ok);

    /* ---- C7 scope block statuses equal the statements' */
    {
        jv *insc = jget(jget(J, "scope"), "in_scope"); int ok = 1; char why[200] = "";
        for (size_t i = 0; i < jlen(insc); i++) {
            jv *e = jidx(insc, i); jv *s = find_statement(STM, jstr(jget(e, "id")) ? jstr(jget(e, "id")) : "");
            if (!s || !jstreq(jget(e, "status"), jstr(jget(s, "status")) ? jstr(jget(s, "status")) : "")) { ok = 0; snprintf(why, sizeof why, "%s: scope says %s, statement says %s", jstr(jget(e, "id")), jstr(jget(e, "status")), s ? jstr(jget(s, "status")) : "(absent)"); }
        }
        if (ok) add("PASS", "C7.scope_statuses", "%zu in-scope statuses agree with the statements", jlen(insc));
        else add("FAIL", "C7.scope_statuses", "%s", why);
    }
    fesetround(FE_TONEAREST);

    /* ---- output */
    const char *verdict = n_fail ? "FAIL" : "PASS";
    buf rep = {0};
    buf_printf(&rep, "cev %s: %s\n", CEV_VERSION, cert_path);
    for (int i = 0; i < NCH; i++) buf_printf(&rep, "%-14s %-26s %s\n", CH[i].st, CH[i].id, CH[i].detail);
    buf_printf(&rep, "verdict: %s (pass %d, fail %d, skip %d, reproduced %d, not reproduced %d, info %d)\n", verdict, n_pass, n_fail, n_skip, n_repro, n_norepro, n_info);
    if (!quiet) fputs(rep.p, stdout);
    if (report_path) { FILE *f = fopen(report_path, "wb"); if (!f) die("cannot write %s", report_path); fwrite(rep.p, 1, rep.n, f); fclose(f); }
    if (json_path) {
        buf jb = {0};
        buf_puts(&jb, "{\"verifier\":\"cev\",\"version\":\"" CEV_VERSION "\",\"certificate\":"); json_escape(&jb, cert_path);
        buf_printf(&jb, ",\"verdict\":\"%s\",\"exit_code\":%d,\"counts\":{\"pass\":%d,\"fail\":%d,\"skip\":%d,\"reproduced\":%d,\"not_reproduced\":%d,\"info\":%d},\"checks\":[",
                   verdict, n_fail ? 1 : 0, n_pass, n_fail, n_skip, n_repro, n_norepro, n_info);
        for (int i = 0; i < NCH; i++) {
            if (i) buf_putc(&jb, ',');
            buf_puts(&jb, "{\"id\":"); json_escape(&jb, CH[i].id);
            buf_puts(&jb, ",\"status\":"); json_escape(&jb, CH[i].st);
            buf_puts(&jb, ",\"detail\":"); json_escape(&jb, CH[i].detail);
            buf_putc(&jb, '}');
        }
        buf_puts(&jb, "]}\n");
        FILE *f = fopen(json_path, "wb"); if (!f) die("cannot write %s", json_path); fwrite(jb.p, 1, jb.n, f); fclose(f);
    }
    return n_fail ? 1 : 0;
}

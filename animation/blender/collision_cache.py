"""Daemon-owned, bounded reuse of evaluated pair collision results.

**Pair-level, not frame-level.** A pair result depends on exactly two things, the two local meshes and their
relative pose, which is `collision.pair_key`, computed by the frame loop from digests memoized per unique mesh. Keying
on it costs nothing extra and hits whenever those two parts are in that relative pose, whatever else moved. A
frame-level key over every body's geometry and placement would change for every frame when one part moves, and
would hash each body's arrays per frame to build.

Deliberately **not** in the namespace:

  * `scene` / `catalog` / `baseline`: a pair verdict is geometry, not scene membership, so two runs sharing a mesh
    pair may share its result.
  * Collision allowances: `verify.allowed` and `InheritedAllowances.allows` are evaluated before the lookup, so a
    verdict is allowance-independent, and including them would empty the cache on every allowance edit.

What is in it: the hashes of the modules that decide a verdict, so a changed tolerance, erosion policy, or Coal
call invalidates every entry.

A project's checks share one on-disk store (`PairStore`) that every worker reads and writes, uncapped: a large film
needs hundreds of thousands of pairs in one pass, and a capped cache shipped inside each check request would let a
repeat compile reuse nothing while the request grew to tens of megabytes.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3

#: Entries are small (one dict per colliding pair, `None` for a clear pair), so
#: this bounds memory rather than rationing a scarce resource. Applies to the
#: in-memory form only (`entries`), which the native session still persists.
MAX_ENTRIES = 50000

#: Bytes of entries the on-disk store keeps before its oldest go. About 100
#: bytes an entry, so ten million pairs; the dog film needs under a million.
STORE_BYTES = 1024 ** 3

#: Pairs a worker keeps in memory in front of the store: every static pair
#: repeats on every frame, and those should not each cost a lookup.
FRONT_ENTRIES = 200000

SCHEMA = 2


def _implementation_digest():
    here = Path(__file__).parent
    digests = {}
    for name in ('collision_cache', 'collision', 'coal_worker', '_mesh_erosion'):
        path = here / (name + '.py')
        if not path.exists():
            path = here.parents[1] / 'verify' / (name + '.py')
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


class PairStore:
    """Pair verdicts on disk, shared by every check of one project.

    SQLite with its rollback journal rather than WAL: workers run in their own
    mount namespaces and projects may sit on network filesystems, where WAL's
    shared-memory index does not hold. Concurrent checks write their new pairs
    in one transaction each at the end, and SQLite's lock orders them, which
    is all "merged under concurrent checks" needs. Reads are point lookups.

    Bounded by bytes: past `STORE_BYTES` the oldest-written pairs go. A
    namespace change (any change to the modules that decide a verdict) empties
    it. Correctness never depends on it, so any SQLite error, or a value it
    cannot read, turns it off for the rest of the call rather than failing the
    check.

    Keys are digested with the namespace: a process that opened the store
    under an older implementation and writes after a newer one emptied it
    writes keys the newer one never looks up.
    """

    def __init__(self, path, namespace):
        self.namespace = namespace.encode() + b':'
        self.db = sqlite3.connect(str(path), timeout=300, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
        # A rowid table, so `INSERT OR REPLACE` gives a rewritten pair a new
        # rowid and eviction by rowid takes the least recently written.
        self.db.execute('CREATE TABLE IF NOT EXISTS pairs (key BLOB UNIQUE NOT NULL, hit TEXT)')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute("SELECT value FROM meta WHERE key = 'namespace'").fetchone()
            if row is None or row[0] != namespace:
                self.db.execute('DELETE FROM pairs')
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('namespace', ?)", (namespace,))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def digest(self, encoded):
        return hashlib.blake2b(self.namespace + encoded.encode(), digest_size=16).digest()

    def get(self, encoded):
        """`(hit, True)` if the store has the pair, else `(None, False)`."""
        row = self.db.execute('SELECT hit FROM pairs WHERE key = ?',
                              (self.digest(encoded),)).fetchone()
        if row is None:
            return None, False
        if row[0] is None:
            return None, True
        hit = json.loads(row[0])
        if not isinstance(hit, dict):
            raise ValueError('not a pair verdict')
        return hit, True

    def put_many(self, items):
        """Write `[(encoded key, hit)]` in one transaction, then keep the store bounded."""
        if not items:
            return
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.executemany('INSERT OR REPLACE INTO pairs (key, hit) VALUES (?, ?)',
                                [(self.digest(key), None if hit is None else json.dumps(hit))
                                 for key, hit in items])
            pages, free, size = (self.db.execute(f'PRAGMA {name}').fetchone()[0]
                                 for name in ('page_count', 'freelist_count', 'page_size'))
            used = (pages - free) * size
            if used > STORE_BYTES:
                # The oldest-written rows, enough of them to land a tenth
                # under the bound; freed pages are reused, so the file stops
                # growing there.
                count = self.db.execute('SELECT count(*) FROM pairs').fetchone()[0]
                drop = min(count, int(count * (1 - STORE_BYTES / used + .1)) + 1)
                self.db.execute('DELETE FROM pairs WHERE rowid <= (SELECT rowid FROM pairs '
                                'ORDER BY rowid LIMIT 1 OFFSET ?)', (drop - 1,))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def close(self):
        self.db.close()


class PairCollisionCache:
    """`entries` is the daemon-owned dict to persist into; `store` a `PairStore` path.

    With a store, pairs are read from it and the call's new pairs are written
    to it by `flush`, which `verify.measure` calls once at the end; `entries`
    is then unused. Without one, `entries` is the in-memory form the native
    session still persists between calls.

    `None` for both means "do not persist" — **not** "do not cache". This
    replaces the frame loop's own per-call `OrderedDict` as well as the
    cross-job cache, so disabling it outright would re-query pairs that repeat
    *within* one call, which is the majority of the reuse on a scene where most
    parts hold still. A caller that passes nothing gets a private dict for the
    duration.
    """

    def __init__(self, entries, *, method, store=None):
        self.hits = self.misses = self.store_errors = 0
        namespace = hashlib.sha256(json.dumps(
            {'schema': SCHEMA, 'method': method,
             'implementation': _implementation_digest()},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.store, self.pending = None, []
        if store is not None:
            try:
                self.store = PairStore(store, namespace)
            except sqlite3.Error:
                self.store_errors += 1
        if self.store is not None or entries is None:
            self.pairs = OrderedDict()
            return
        # A namespace change means a different verdict function; drop rather
        # than keep entries a new implementation would answer differently.
        if entries.get('namespace') != namespace:
            entries.clear()
            entries['namespace'] = namespace
            entries['pairs'] = {}
        self.pairs = entries.setdefault('pairs', {})

    @staticmethod
    def encode(key):
        """`collision.pair_key`'s tuple as one JSON-safe string.

        `Body.key` is a raw digest, so the tuple cannot be a JSON key as-is.
        """
        a, b, bins = key
        return a.hex() + b.hex() + ':' + ','.join(map(str, bins))

    def get(self, key):
        """`(hit, True)` on a hit; `(None, False)` on a miss.

        A two-tuple rather than a sentinel because `None` is a *legitimate*
        cached value — it is what a clear pair records, and the whole point is
        to not re-run the query for pairs already known to be clear.
        """
        encoded = self.encode(key)
        if encoded in self.pairs:
            if self.store is not None:
                self.pairs.move_to_end(encoded)
            hit = self.pairs[encoded]
        elif self.store is not None:
            try:
                hit, found = self.store.get(encoded)
            except (sqlite3.Error, ValueError):
                self.store_errors += 1
                self.store, found = None, False
            if not found:
                self.misses += 1
                return None, False
            self._remember(encoded, hit)
        else:
            self.misses += 1
            return None, False
        self.hits += 1
        return (dict(hit) if hit is not None else None), True

    def put(self, key, hit):
        encoded, hit = self.encode(key), (dict(hit) if hit is not None else None)
        if self.store is not None:
            self.pending.append((encoded, hit))
            self._remember(encoded, hit)
            return
        self.pairs[encoded] = hit
        while len(self.pairs) > MAX_ENTRIES:
            del self.pairs[next(iter(self.pairs))]

    def get_value(self, name):
        """`(value, True)` for a measured value stored under `name`, else `(None, False)`.

        Values -- the overlap diagnostics -- share the pair store, under keys
        their callers make exact: a value is reused only for the very inputs
        it was measured on.
        """
        encoded = 'v:' + name
        if encoded in self.pairs:
            return self.pairs[encoded]['v'], True
        if self.store is not None:
            try:
                held, found = self.store.get(encoded)
            except (sqlite3.Error, ValueError):
                self.store_errors += 1
                self.store, found = None, False
            if found and isinstance(held, dict) and 'v' in held:
                self._remember(encoded, held)
                return held['v'], True
        return None, False

    def put_value(self, name, value):
        """Store a value with the pair store; without one there is nothing to reuse it across."""
        if self.store is None:
            return
        encoded, held = 'v:' + name, {'v': value}
        self.pending.append((encoded, held))
        self._remember(encoded, held)

    def _remember(self, encoded, hit):
        self.pairs[encoded] = hit
        if len(self.pairs) > FRONT_ENTRIES:
            self.pairs.popitem(last=False)

    def flush(self):
        """Write this call's new pairs to the store, if there is one."""
        if self.store is None:
            return
        try:
            self.store.put_many(self.pending)
        except sqlite3.Error:
            self.store_errors += 1
        self.pending = []
        self.store.close()
        self.store = None

    def stats(self):
        return {'collision_pair_cache_hits': self.hits,
                'collision_pair_cache_misses': self.misses,
                'collision_pair_store_errors': self.store_errors}

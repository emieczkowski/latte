from pathlib import Path

class LockManager:
    def __init__(self, lock_dir):
        self.lock_dir = lock_dir
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def _lock_path(self, rel_path):
        safe = rel_path.replace("/", "__")
        return self.lock_dir / f"{safe}.lock"

    def acquire(self, rel_path, owner):
        p = self._lock_path(rel_path)
        try:
            fd = p.open("x")
            fd.write(owner)
            fd.close()
            return True
        except FileExistsError:
            return False

    def release(self, rel_path):
        p = self._lock_path(rel_path)
        if p.exists():
            p.unlink()

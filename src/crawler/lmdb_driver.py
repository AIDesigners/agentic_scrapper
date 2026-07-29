import lmdb
import os
from typing import Optional, Iterable, Set

class LMDB(object):
    def __init__(self, db_path : str = "./data/lmdb.db", map_size : int = 104857600, readonly : bool = False) -> None :
        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size, subdir=False, readonly=readonly, metasync=False,
                             max_readers=0xF, max_dbs=0, lock=True)
    def put(self, key : str, value : str) -> bool :
        status = False
        try :
            with self.env.begin(write=True) as txn:
                txn.put(key.encode('utf-8'), value.encode('utf-8'))
                status = True
        except lmdb.MapFullError :
            print("Error: Database map size is full!")
        except lmdb.Error as e :
            print(f"An unexpected LMDB error occurred: {e}")
        return status
    def get(self, key : str) -> Optional[str] :
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode('utf-8'))
            return value.decode('utf-8') if value else None
    def check_keys(self, keys : Iterable[str]) -> Set[str] :
        new_keys = set()
        with self.env.begin(write=False) as txn:
            for key in keys:
                if txn.get(str(key).encode('utf-8')) is None:
                    new_keys.add(key)
        return new_keys
    def close(self) -> None :
        if self.env :
            self.env.close()
    # Context manager
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


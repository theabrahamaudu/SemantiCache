import glob
import time
import tempfile
from semanticache import Cache


query_str = "How many bananas do we need?"
similar_query_str = "How many bananas do we require?"
different_query_str = "How many bananas do we have?"
response_str = "this is a very cool response"

temp_dir = tempfile.TemporaryDirectory()

cache = Cache(
    cache_path=temp_dir.name+"/sem_cache",
    config_path=temp_dir.name+"/sem_config"
)


def test_clear_cache():
    cache.set(query_str, response_str)
    cache.clear()
    assert len(cache.cache_index.index_to_docstore_id) == 0  # type: ignore


def test_clear_cache_and_files():
    cache.clear(clear_files=True)
    assert len(glob.glob(cache.cache_path+"/*")) == 0
    assert cache.cache_index is None


def test_trim_cache_by_size():
    set_size = 3
    with tempfile.TemporaryDirectory() as tmp:
        c = Cache(
            cache_size=set_size,
            cache_path=tmp+"/sem_cache",
            config_path=tmp+"/sem_config"
        )
        for q in range(5):
            c.set(str(q), response_str)
        assert len(
            c.cache_index.index_to_docstore_id.items()  # type: ignore
        ) == set_size


def test_trim_cache_by_ttl():
    ttl = 10
    with tempfile.TemporaryDirectory() as tmp:
        c = Cache(
            ttl=ttl,
            trim_by_size=False,
            cache_path=tmp+"/sem_cache",
            config_path=tmp+"/sem_config"
        )
        for q in range(5):
            c.set(str(q), response_str)
        time.sleep(1.5*ttl)
        c.set("random", response_str)
        assert len(
            c.cache_index.index_to_docstore_id.items()  # type: ignore
        ) == 1


def test_update_leaderboard():
    cache.set(query_str, response_str)
    cache.set(similar_query_str, response_str)
    hits = 3
    for _ in range(hits):
        _ = cache.get(query_str)
    _ = cache.get(similar_query_str)

    leaderboard: list = cache.read_leaderboard()  # type: ignore
    assert leaderboard[0]["hits"] == hits


def test_truncate_leaderboard():
    hits = 7
    for q in range(hits):
        cache.set(str(q), response_str)
        for _ in range(q):
            _ = cache.get(str(q))
    leaderboard: list = cache.read_leaderboard()  # type: ignore
    assert len(leaderboard) == cache.leaderboard_top_n

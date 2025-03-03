import tempfile
import pytest
from semanticache import SemantiCache as Cache


query_str = "How many bananas do we need?"
similar_query_str = "How many bananas do we require?"
different_query_str = "How many bananas do we have?"
response_str = "this is a very cool response"

temp_dir = tempfile.TemporaryDirectory()


def test_create_cache():
    try:
        _ = Cache(
                cache_path=temp_dir.name+"/sem_cache",
                config_path=temp_dir.name+"/sem_config"
            )
    except Exception as e:
        pytest.fail(str(e))


cache = Cache(
    cache_path=temp_dir.name+"/sem_cache",
    config_path=temp_dir.name+"/sem_config"
)


def test_set_value():
    try:
        cache.set(
            query=query_str,
            response=response_str
        )
    except Exception as e:
        pytest.fail(str(e))


def test_get_value():
    try:
        _ = cache.get(query_str)
    except Exception as e:
        pytest.fail(str(e))


def test_same_query_same_response():
    cache_resp = cache.get(query_str)
    assert cache_resp == response_str


def test_similar_query_same_response():
    cache_resp = cache.get(similar_query_str)
    assert cache_resp == response_str


def test_different_query_no_response():
    cache_resp = cache.get(different_query_str)
    assert cache_resp is None

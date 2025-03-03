![Coverage](https://img.shields.io/badge/coverage-75%25-brightgreen)

# SemantiCache

SemantiCache is a semantic caching library that enables efficient storage and retrieval of query-response pairs using FAISS and vector embeddings. It supports cache expiration, trimming strategies, and query similarity search.

## Features
- **Vector-based caching** using FAISS and HuggingFace embeddings
- **Query similarity search** for retrieving semantically relevant responses
- **Automatic cache management** with size and TTL-based trimming
- **Leaderboard tracking** for frequently accessed queries
- **Persistent storage** for cache state management

## Installation

```sh
pip install semanticache
```

## Usage

### Full Documentation
Read the [docs](https://github.com/theabrahamaudu/SemantiCache/blob/main/docs/SemantiCacheDocs.md)
_________________________________________________________________________

### Initializing SemantiCache

```python
from semanticache import Cache

cache = Cache(
    trim_by_size=True,
    cache_path="./sem_cache",
    config_path="./sem_config",
    cache_size=100,
    ttl=3600,
    threshold=0.1,
    leaderboard_top_n=5,
    log_level="INFO"
)
```

### Storing a Query-Response Pair

```python
cache.set("What is the capital of France?", "Paris")
```

### Retrieving a Cached Response

```python
response = cache.get("What is the capital of France?")
print(response)  # Output: Paris
```

### Clearing the Cache

```python
cache.clear(clear_files=True)  # Clears all cache entries and files
```

### Reading the Leaderboard

```python
leaderboard = cache.read_leaderboard()
print(leaderboard)  # Outputs most frequently accessed queries
```

## Configuration
SemantiCache stores configuration parameters in `./sem_config/sem_config.yaml`. These include:

```yaml
cache:
  path: ./sem_cache
  name: sem_cache_index
  cache_size: 100
  ttl: 3600
  threshold: 0.1
  trim_by_size: True
  leaderboard_top_n: 5
```

## Example: Using the Cache with an LLM

This example demonstrates how to check the cache before querying an LLM and how to store responses when needed.

```python
from semanticache import Cache

def call_llm(query):
    """Simulate an LLM call (replace with actual API call)."""
    return f"Response for: {query}"

# Initialize the cache
cache = Cache()

# Example query
query = "What is the capital of France?"

# Check if the response is already cached
cached_response = cache.get(query)

if cached_response:
    print(f"Cache Hit: {cached_response}")
else:
    print("Cache Miss: Querying LLM...")
    response = call_llm(query)
    cache.set(query, response)
    print(f"Stored in cache: {response}")
```

## Advanced Settings
### Cache Trimming
- **By Size:** Keeps the most accessed entries up to `cache_size`
- **By Time (TTL):** Removes entries older than `ttl` seconds

    This is toggled by setting `trim_by_size` to `True` or `False` in config file or during initialization in script

### Similarity Threshold
- Determines when a query matches an existing cache entry
- A lower threshold increases exact matches, while a higher one allows more flexible retrieval

## Dependencies
- `FAISS` for vector similarity search
- `HuggingFace` from `Langchain Community` for embedding generation
- `yaml`, `numpy`, `json`, and `pickle` for serialization

## Help
Feel free to reach out to me or create a new issue if you encounter any problems using SemantiCache

## Contribution: Possible Improvements/Ideas

- [ ] More unit tests
- [ ] Less dependence on other libraries
- [ ] Support for alternate vector index engines like ChromaDB, Milvus, etc.
- [ ] More optimized logic where possible
- [ ] Implement more sophisticated ranking and pruning algorithms.
- [ ] Support additional embedding models for improved semantic search.

## Authors

Contributors names and contact info

*Abraham Audu*

* GitHub - [@the_abrahamaudu](https://github.com/theabrahamaudu)
* X (formerly Twitter) - [@the_abrahamaudu](https://x.com/the_abrahamaudu)
* LinkedIn - [@theabrahamaudu](https://www.linkedin.com/in/theabrahamaudu/)
* Instagram - [@the_abrahamaudu](https://www.instagram.com/the_abrahamaudu/)
* YouTube - [@DataCodePy](https://www.youtube.com/@DataCodePy)

## Version History

* See [commit change](https://github.com/theabrahamaudu/SemantiCache/commits/main/)
* See [release history](https://github.com/theabrahamaudu/SemantiCache/releases)

## Acknowledgments

* This library was built on top of `FAISS`, `HuggingFace` and `LangChain`

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.



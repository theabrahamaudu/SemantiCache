# SemantiCache Documentation

## Overview

**SemantiCache** is an intelligent semantic caching system that efficiently stores and retrieves query-response pairs using similarity search powered by FAISS (Facebook AI Similarity Search). It optimizes caching by automatically trimming entries based on size or time-to-live (TTL) and tracks popular queries via a leaderboard.

---

## Public Methods

### `get(query: str) -> str | None`
Retrieves a cached response for a given query.

- **Behavior:**  
  Searches for a record using semantic similarity. If a matching record is found and its similarity score meets the threshold, it updates the record's metadata (hit count and update timestamp) and refreshes the leaderboard, then returns the cached response. Otherwise, it returns `None`.

### `set(query: str, response: str) -> None`
Stores a new query-response pair in the cache.

- **Behavior:**  
  Creates a new record with the current timestamp, adds it to the cache, and triggers a cache trim to enforce size or TTL constraints.

### `clear(clear_files: bool = False) -> None`
Clears all cached records.

- **Behavior:**  
  Removes all records from the cache index. If `clear_files` is set to True, it also deletes all files in the cache directory and resets the cache index.

### `read_leaderboard() -> list | None`
Reads and returns the leaderboard data.

- **Behavior:**  
  Loads the leaderboard from a JSON file, which contains the top queries based on access frequency. Returns `None` if an error occurs during file read.

---

## Private/Internal Methods

These methods are intended for internal use to support the public API:

- **`__trim_cache() -> None`**  
  Trims the cache by removing either the least-hit records (if trimming by size) or outdated records (if trimming by TTL).

- **`__create_record(created_at: datetime, query: str, response: str, updated_at: datetime | None = None, record_id: str | None = None, hits: int | None = None) -> str | None`**  
  Creates and stores a new cache record with metadata, such as timestamps and hit counts.

- **`__read_record(text: str) -> Document | None`**  
  Retrieves a record from the cache by performing a semantic similarity search using FAISS.

- **`__update_record(record: Document, updated_at: datetime) -> None`**  
  Updates an existing record by incrementing its hit count and refreshing its update timestamp. The old record is removed and replaced with an updated version.

- **`__delete_record(id: str) -> None`**  
  Deletes a record from the cache by its unique identifier and persists the updated cache state.

- **`remove(vectorstore: FAISS, docstore_ids: list[str] | None) -> tuple[int, int]`**  
  A static method to remove specified records (or all records) from the FAISS vector store and update internal mappings.

- **`__update_leaderboard(query: str, response: str, hits: int) -> None`**  
  Updates the leaderboard JSON file with the latest hit counts for queries, ensuring it contains only the top N entries.

- **`__get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu", normalize_embeddings: bool = False) -> HuggingFaceEmbeddings`**  
  Initializes and returns a HuggingFace embeddings model used for semantic similarity searches.

- **`__configure_cache_index(query: str, response: str, new_id: str) -> FAISS | None`**  
  Creates and configures a new FAISS cache index with an initial record.

- **`__load_cache_index() -> FAISS`**  
  Loads the FAISS cache index from local persistent storage.

- **`__persist_cache_state() -> None`**  
  Saves the current state of the cache index to local storage.

- **`__check_params() -> None`**  
  Validates and sets default values for configuration parameters, overriding them if necessary based on the server configuration.

- **`__load_config() -> Dict[str, Any]`**  
  Loads configuration settings from a YAML file. If the file is not found, it creates the necessary directories and a default configuration file.

- **`__create_directories() -> None`**  
  Creates the required cache and configuration directories if they do not exist.

- **`__create_config_file() -> None`**  
  Creates a default YAML configuration file with predefined caching settings if it does not already exist.

---

## Configuration

SemantiCache settings are stored in a YAML configuration file (typically located at `./sem_config/sem_config.yaml`) and include parameters such as:

```yaml
cache:
    path: ./sem_cache
    name: sem_cache_index
    cache_size: 100
    ttl: 3600
    threshold: 0.1
    trim_by_size: true
    leaderboard_top_n: 5
```

These parameters control aspects like cache size, time-to-live for entries, similarity thresholds, and leaderboard size.

---

## Usage Example

Below is a basic example of how to use SemantiCache:

```python
from semantichache import SemantiCache

# Initialize the cache
semantic_cache = SemantiCache()

# Attempt to retrieve a cached response
query = "What is artificial intelligence?"
response = semantic_cache.get(query)

# If no cached response is found, generate and store one
if response is None:
    response = "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."
    semantic_cache.set(query, response)

print(response)
```

---

## Logging

SemantiCache logs key operations such as cache reads, writes, updates, and errors. This information is useful for debugging and understanding cache performance.

---

## Possible Enhancements

- **Distributed Caching:** Extend support for distributed systems.
- **Advanced Ranking:** Implement more sophisticated ranking and pruning algorithms.
- **Model Integrations:** Support additional embedding models for improved semantic search.


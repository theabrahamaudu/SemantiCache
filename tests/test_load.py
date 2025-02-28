from semanticache import Cache

cache = Cache()

cache.set("test", "test response")
ans =cache.get("test")
print(ans)

import time
from semanticache import Cache

start = time.perf_counter()
cache = Cache(ttl=65, trim_by_size=False, log_level="WARNING")
end = time.perf_counter() - start
print("cache load time:: %s secs" % round(end, 2))


def run(iter: int):
    start = time.perf_counter()
    cache.set("test", "test response")
    ans = cache.get("test")
    end = time.perf_counter() - start
    print(ans, f"iter {iter} run time: {round(end, 2)} secs")
    return end


run_times = []
for iter in range(10):
    run_times.append(run(iter))
print("average run time %s secs" % round(sum(run_times)/len(run_times), 2))

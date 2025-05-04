import functools
import hashlib
import logging
from threading import RLock

import time

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache with time-based expiration."""

    def __init__(self, max_size=128, ttl=3600):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items in cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.usage_order = []
        self._lock = RLock()

    def get(self, key):
        """Get item from cache if it exists and is not expired."""
        with self._lock:
            if key not in self.cache:
                return None

            value, timestamp = self.cache[key]
            current_time = time.time()

            # Check if item is expired
            if current_time - timestamp > self.ttl:
                # Remove expired item
                del self.cache[key]
                self.usage_order.remove(key)
                return None

            # Update usage order
            self.usage_order.remove(key)
            self.usage_order.append(key)

            return value

    def set(self, key, value):
        """Add item to cache with current timestamp."""
        with self._lock:
            # If key already exists, update usage order
            if key in self.cache:
                self.usage_order.remove(key)

            # If cache is full, evict least recently used item
            if len(self.cache) >= self.max_size and len(self.usage_order) > 0:
                lru_key = self.usage_order.pop(0)
                del self.cache[lru_key]

            # Add new item
            self.cache[key] = (value, time.time())
            self.usage_order.append(key)

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.usage_order.clear()


def ttl_cache(maxsize=128, ttl=3600):
    """
    Decorator that caches function results with a time-to-live (TTL).

    Args:
        maxsize: Maximum cache size
        ttl: Time to live in seconds
    """
    cache = LRUCache(max_size=maxsize, ttl=ttl)
    lock = RLock()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from the function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            with lock:
                result = cache.get(key)
                if result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result

                # Call the function
                result = func(*args, **kwargs)

                # Store in cache
                cache.set(key, result)
                return result

        # Add clear_cache method
        def clear_cache():
            with lock:
                cache.clear()

        wrapper.clear_cache = clear_cache
        return wrapper

    return decorator

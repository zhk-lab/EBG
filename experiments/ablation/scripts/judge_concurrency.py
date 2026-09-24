"""Share a request limit across judge workers and back off on HTTP 429."""
import re
import time
from threading import Condition


class AdaptiveConcurrency:
    def __init__(self, initial=5, maximum=32, cooldown=60, stable_seconds=60,
                 successes_to_grow=20, clock=time.monotonic, on_change=None):
        self.limit = initial
        self.maximum = maximum
        self.cooldown = cooldown
        self.stable_seconds = stable_seconds
        self.successes_to_grow = successes_to_grow
        self.clock = clock
        self.on_change = on_change
        self.condition = Condition()
        self.active = 0
        self.blocked_until = 0
        self.changed_at = clock()
        self.successes = 0
        self.rate_limits = 0

    def snapshot(self):
        return dict(limit=self.limit, active=self.active, rate_limits=self.rate_limits)

    def acquire(self):
        with self.condition:
            while self.active >= self.limit or self.clock() < self.blocked_until:
                delay = max(0, self.blocked_until - self.clock())
                self.condition.wait(timeout=min(delay, 1) if delay else 1)
            self.active += 1

    def release(self, *, success=False, rate_limited=False, retry_after=0):
        with self.condition:
            self.active -= 1
            now = self.clock()
            event = None
            if rate_limited:
                self.rate_limits += 1
                self.successes = 0
                if now >= self.blocked_until:
                    self.limit = max(1, self.limit // 2)
                    event = 'rate_limited'
                self.blocked_until = max(self.blocked_until, now + max(self.cooldown, retry_after))
                self.changed_at = now
            elif success and now >= self.blocked_until:
                self.successes += 1
                if (self.successes >= self.successes_to_grow
                        and now - self.changed_at >= self.stable_seconds and self.limit < self.maximum):
                    self.limit = min(self.maximum, self.limit + max(1, self.limit // 2))
                    self.changed_at = now
                    self.successes = 0
                    event = 'increase'
            if event and self.on_change:
                self.on_change(dict(event=event, **self.snapshot()))
            self.condition.notify_all()

    def wrap(self, client):
        gate = self

        class LimitedClient:
            @property
            def profile(self):
                return client.profile

            def complete(self, *args, **kwargs):
                gate.acquire()
                try:
                    result = client.complete(*args, **kwargs)
                except BaseException as error:
                    limited = re.search(r'\b429\b', str(error)) is not None
                    cause = error.__cause__
                    delay = 0
                    if limited and getattr(cause, 'headers', None):
                        try:
                            delay = float(cause.headers.get('Retry-After', '0'))
                        except (TypeError, ValueError):
                            pass
                    gate.release(rate_limited=limited, retry_after=delay)
                    raise
                gate.release(success=True)
                return result

        return LimitedClient()

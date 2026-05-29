"""Script to add queue classes to lb_proxy.py."""

# Read the current file
with open("src/llamacpp_cli/lb_proxy.py") as f:
    content = f.read()

# Queue classes to insert
queue_classes = '''

@dataclass
class QueuedRequest:
    """A request waiting in the queue."""

    request: Request
    model: str | None
    enqueued_at: float
    future: asyncio.Future = field(default_factory=asyncio.Future)


@dataclass
class RequestQueue:
    """Request queue with backpressure control."""

    max_size: int = 1000
    timeout: float = 60.0
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    total_queued: int = 0
    total_timeouts: int = 0
    total_rejections: int = 0
    wait_times: list[float] = field(default_factory=list)

    async def enqueue(self, request: Request, model: str | None) -> QueuedRequest:
        """Add a request to the queue. Raises HTTPException if queue is full."""
        if self.queue.qsize() >= self.max_size:
            self.total_rejections += 1
            # Estimate wait time based on recent performance
            avg_wait = sum(self.wait_times[-100:]) / len(self.wait_times[-100:]) if self.wait_times else 0
            estimated_wait = avg_wait * self.queue.qsize()
            raise HTTPException(
                status_code=503,
                detail=f"Queue full ({self.max_size} requests). Estimated wait: {estimated_wait:.1f}s",
            )

        queued = QueuedRequest(request=request, model=model, enqueued_at=time.time())
        await self.queue.put(queued)
        self.total_queued += 1
        return queued

    async def dequeue(self) -> QueuedRequest:
        """Get the next request from the queue."""
        return await self.queue.get()

    def size(self) -> int:
        """Current queue depth."""
        return self.queue.qsize()

    def record_wait_time(self, wait_time: float) -> None:
        """Record a wait time for statistics."""
        self.wait_times.append(wait_time)
        # Keep only last 1000 entries
        if len(self.wait_times) > 1000:
            self.wait_times = self.wait_times[-1000:]

    def get_percentiles(self) -> dict[str, float]:
        """Calculate wait time percentiles."""
        if not self.wait_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_times = sorted(self.wait_times)
        n = len(sorted_times)
        return {
            "p50": sorted_times[int(n * 0.5)] if n > 0 else 0.0,
            "p95": sorted_times[int(n * 0.95)] if n > 1 else 0.0,
            "p99": sorted_times[int(n * 0.99)] if n > 2 else 0.0,
        }
'''

# Find the position to insert (after RateLimiter class, before Backend class)
insertion_point = content.find("\n\n@dataclass\nclass Backend:")

if insertion_point == -1:
    print("ERROR: Could not find Backend class")
    exit(1)

# Insert the queue classes
new_content = content[:insertion_point] + queue_classes + content[insertion_point:]

# Write the modified content
with open("src/llamacpp_cli/lb_proxy.py", "w") as f:
    f.write(new_content)

print("Successfully added queue classes")

"""Tests for circuit breaker implementation in lb-proxy."""

import time

import pytest

from llamacpp_cli.lb_proxy import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    """Test circuit breaker state transitions and behavior."""

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_attempt_request() is True

    def test_closed_to_open_after_threshold_failures(self):
        """Circuit should open after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures
        for _ in range(2):
            cb.record_failure()
            assert cb.state == CircuitState.CLOSED

        # Third failure should open the circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_attempt_request() is False

    def test_success_resets_failure_count_in_closed_state(self):
        """Successful request should reset failure count in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_success()
        assert cb.failure_count == 0

        # Now need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_open_to_half_open_after_timeout(self):
        """Circuit should transition to HALF_OPEN after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Immediate request should be rejected
        assert cb.can_attempt_request() is False

        # Wait for timeout
        time.sleep(0.15)

        # Should now be HALF_OPEN
        assert cb.can_attempt_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_after_success_threshold(self):
        """Circuit should close after success threshold in HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=2, success_threshold=2, timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait and transition to HALF_OPEN
        time.sleep(0.15)
        cb.can_attempt_request()
        assert cb.state == CircuitState.HALF_OPEN

        # First success
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        # Second success should close circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Any failure in HALF_OPEN should reopen the circuit."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait and transition to HALF_OPEN
        time.sleep(0.15)
        cb.can_attempt_request()
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_attempt_request() is False

    def test_half_open_timeout_reopens_circuit(self):
        """Circuit should reopen if HALF_OPEN timeout expires."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1, half_open_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait and transition to HALF_OPEN
        time.sleep(0.15)
        cb.can_attempt_request()
        assert cb.state == CircuitState.HALF_OPEN

        # Wait for half-open timeout
        time.sleep(0.15)

        # Should reopen
        assert cb.can_attempt_request() is False
        assert cb.state == CircuitState.OPEN

    def test_get_state_info(self):
        """get_state_info should return comprehensive state."""
        cb = CircuitBreaker(failure_threshold=3)

        info = cb.get_state_info()
        assert info["state"] == "closed"
        assert info["failure_count"] == 0
        assert info["total_opens"] == 0
        assert info["total_closes"] == 0

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        info = cb.get_state_info()
        assert info["state"] == "open"
        assert info["total_opens"] == 1
        assert info["seconds_until_retry"] is not None
        assert info["seconds_since_last_failure"] is not None

    def test_total_opens_and_closes_tracking(self):
        """Circuit should track total opens and closes."""
        cb = CircuitBreaker(failure_threshold=2, success_threshold=2, timeout=0.1)

        # Open circuit (first time)
        cb.record_failure()
        cb.record_failure()
        assert cb.total_opens == 1

        # Close circuit
        time.sleep(0.15)
        cb.can_attempt_request()  # Transition to HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.total_closes == 1
        assert cb.state == CircuitState.CLOSED

        # Open again
        cb.record_failure()
        cb.record_failure()
        assert cb.total_opens == 2

        # Close again
        time.sleep(0.15)
        cb.can_attempt_request()
        cb.record_success()
        cb.record_success()
        assert cb.total_closes == 2

    def test_custom_thresholds(self):
        """Circuit breaker should respect custom thresholds."""
        cb = CircuitBreaker(
            failure_threshold=10,
            success_threshold=5,
            timeout=5.0,
            half_open_timeout=2.0,
        )

        # Should not open until 10 failures
        for _ in range(9):
            cb.record_failure()
            assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_mixed_success_and_failure_in_closed_state(self):
        """Circuit should handle mixed success/failure in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Resets failure count
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Only 2 consecutive failures

        cb.record_failure()
        assert cb.state == CircuitState.OPEN  # Now 3 consecutive

    def test_failure_time_tracking(self):
        """Circuit should track last failure time."""
        cb = CircuitBreaker()

        assert cb.last_failure_time == 0.0

        before = time.time()
        cb.record_failure()
        after = time.time()

        assert before <= cb.last_failure_time <= after

    def test_consecutive_successes_in_closed_state(self):
        """Circuit should track consecutive successes in CLOSED state."""
        cb = CircuitBreaker()

        assert cb.success_count == 0

        cb.record_success()
        assert cb.success_count == 1

        cb.record_success()
        assert cb.success_count == 2

        cb.record_failure()
        assert cb.success_count == 0

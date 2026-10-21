"""Two-read caches: certified scalar softmax readouts and range-kernel filters."""
from .scalar import adaptive_nodes, interpolate, interval_envelope, rows
from .bilateral import bilateral_exact, bilateral_two_read, error_bound

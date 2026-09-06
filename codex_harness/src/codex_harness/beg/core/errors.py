"""BEG exception hierarchy."""


class BEGError(ValueError):
    """Base class for deterministic input or graph errors."""


class BundleError(BEGError):
    """A visible bundle violates the input contract."""


class ExtractionError(BEGError):
    """Visible facts cannot be extracted without ambiguity."""


class BehaviorError(BEGError):
    """Atomic behaviors cannot be built or validated."""


class RelationError(BEGError):
    """Behavior relations cannot be built or validated."""


class GraphError(BEGError):
    """The final graph violates its schema-level invariants."""


class DirectoryError(BEGError):
    """A behavior directory cannot be built deterministically."""

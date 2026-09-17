"""EBG exception hierarchy."""


class EBGError(ValueError):
    """Base class for deterministic input or graph errors."""


class BundleError(EBGError):
    """A visible bundle violates the input contract."""


class ExtractionError(EBGError):
    """Visible facts cannot be extracted without ambiguity."""


class BehaviorError(EBGError):
    """Atomic behaviors cannot be built or validated."""


class RelationError(EBGError):
    """Behavior relations cannot be built or validated."""


class GraphError(EBGError):
    """The final graph violates its schema-level invariants."""


class DirectoryError(EBGError):
    """A behavior directory cannot be built deterministically."""

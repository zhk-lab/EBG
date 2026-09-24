"""Shared configuration for Gold review and dataset validation."""

SAMPLES_BY_SWAP_TYPE = {
    "input_validation_boundary": (1, 13, 18, 20, 21, 27, 73, 75, 92, 96),
    "parsing_matching": (3, 4, 9, 29, 33, 34, 38, 39, 40, 42, 47, 49, 51, 72, 77, 79, 91),
    "ordering_precedence": (2, 5, 8, 10, 16, 19, 25, 32, 37, 41, 56, 70, 78, 81, 85, 89, 93, 95, 97),
    "default_null_fallback": (12, 15, 26, 30, 36, 52, 55, 58, 60, 67, 76, 83, 88),
    "exception_error_handling": (14, 17, 28, 43, 65, 66, 68, 84, 94, 99, 100),
    "state_identity_lifecycle": (7, 11, 24, 35, 62, 64, 69, 74, 80, 90),
    "representation_normalization": (6, 22, 31, 46, 53, 59, 71, 82, 86, 87, 98),
    "dispatch_routing_aggregation": (23, 44, 45, 48, 50, 54, 57, 61, 63),
}
SWAP_TYPE_BY_SAMPLE = {
    sample_number: swap_type
    for swap_type, sample_numbers in SAMPLES_BY_SWAP_TYPE.items()
    for sample_number in sample_numbers
}
DOUBLE_REVIEW_SAMPLES = (
    6, 8, 12, 14, 18, 19, 24, 41, 42, 44, 49, 50, 55,
    64, 66, 70, 72, 77, 82, 83, 85, 92, 96, 98, 99,
)
PORTABLE_ROOT = "${SILENTSWAP_ROOT}"

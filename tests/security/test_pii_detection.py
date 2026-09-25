# callcenter-intelligence
# tests/security/test_pii_detection.py

from __future__ import annotations

import pytest

from src.security.pii_redactor import redact_pii

SSN_VARIANTS = ["123-45-6789", "123 45 6789", "123.45.6789", "123456789", "987-65-4321"]
CARD_VARIANTS = [
    "4111-1111-1111-1111",
    "4111 1111 1111 1111",
    "4111111111111111",
    "4111, 1111, 1111, 1111",
    "5500-0000-0000-0004",
    "378282246310005",
]
EMAIL_VARIANTS = [
    "jane@co.com",
    "jane.doe@sub.example.org",
    "j+tag@x.io",
    "JANE@CO.COM",
    "a_b-c@mail.co.uk",
    "jane.do at example.com",
]
PHONE_VARIANTS = [
    "555-123-4567",
    "(555) 123-4567",
    "555.123.4567",
    "+1 555 123 4567",
    "5551234567",
]


@pytest.mark.parametrize("raw", SSN_VARIANTS)
def test_ssn_variants_redacted(raw: str) -> None:
    result = redact_pii(f"My social is {raw} thanks.")
    assert result.pii_found
    assert "[REDACTED_SSN]" in result.redacted_text
    assert raw not in result.redacted_text


@pytest.mark.parametrize("raw", CARD_VARIANTS)
def test_credit_card_variants_redacted(raw: str) -> None:
    result = redact_pii(f"Charge it to {raw} please.")
    assert result.pii_found
    assert "[REDACTED_CREDIT_CARD]" in result.redacted_text
    assert raw not in result.redacted_text


@pytest.mark.parametrize("raw", EMAIL_VARIANTS)
def test_email_variants_redacted(raw: str) -> None:
    result = redact_pii(f"Send confirmation to {raw} today.")
    assert result.pii_found
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert raw not in result.redacted_text


@pytest.mark.parametrize("raw", PHONE_VARIANTS)
def test_phone_variants_redacted(raw: str) -> None:
    result = redact_pii(f"Call me back on {raw} after five.")
    assert result.pii_found
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert raw not in result.redacted_text


def test_spec_ssn_example() -> None:
    assert "[REDACTED_SSN]" in redact_pii("My SSN is 123-45-6789").redacted_text


def test_spec_two_types_one_pass() -> None:
    result = redact_pii("Call 555-123-4567 or email jane@co.com")
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert result.replacements == 2


def test_clean_text_untouched() -> None:
    original = "Hello, how can I help?"
    result = redact_pii(original)
    assert result.pii_found is False
    assert result.redacted_text == original
    assert result.replacements == 0


def test_right_to_left_preserves_later_offsets() -> None:
    """The SSN placeholder is longer than the SSN it replaces. Redacting left-to-right would
    shift every later span by that difference and the email replacement would land off-target."""
    text = "ssn 123-45-6789 mid jane@co.com end"
    result = redact_pii(text)
    assert result.redacted_text == "ssn [REDACTED_SSN] mid [REDACTED_EMAIL] end"


def test_surrounding_text_intact() -> None:
    result = redact_pii("BEFORE 123-45-6789 AFTER")
    assert result.redacted_text.startswith("BEFORE ")
    assert result.redacted_text.endswith(" AFTER")


def test_overlap_resolved_by_priority() -> None:
    """A 16-digit card must redact as CARD, never as four phone-ish fragments."""
    result = redact_pii("card 4111111111111111 done")
    assert result.pii_types == ("CREDIT_CARD",)
    assert result.replacements == 1


def test_many_items_all_redacted() -> None:
    text = "a 123-45-6789 b jane@co.com c 555-123-4567 d 4111-1111-1111-1111 e"
    result = redact_pii(text)
    assert result.replacements == 4
    assert set(result.pii_types) == {"SSN", "EMAIL", "PHONE", "CREDIT_CARD"}


def test_spans_are_sorted_and_non_overlapping() -> None:
    result = redact_pii("x 123-45-6789 y jane@co.com z 555-123-4567")
    starts = [s for s, _, _ in result.spans]
    assert starts == sorted(starts)
    for (_, prev_end, _), (next_start, _, _) in zip(result.spans, result.spans[1:]):
        assert prev_end <= next_start


def test_empty_string() -> None:
    result = redact_pii("")
    assert result.pii_found is False
    assert result.redacted_text == ""


def test_plain_at_without_a_domain_is_not_email() -> None:
    original = "I will look at that later."
    result = redact_pii(original)
    assert result.pii_found is False
    assert result.redacted_text == original


def test_non_string_raises() -> None:
    with pytest.raises(TypeError):
        redact_pii(None)  # type: ignore[arg-type]

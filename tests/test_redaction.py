from narada.redaction import redact_text


def test_redacts_core_personal_information() -> None:
    source = (
        "Alice email is alice@example.com, phone is +1 415-555-1212, "
        "card 4111 1111 1111 1111, SSN 123-45-6789, IP 192.168.0.20, "
        "site https://example.org, address 123 Main Street, "
        "IBAN GB29NWBK60161331926819."
    )
    output = redact_text(source)

    assert "Alice" in output
    assert "[REDACTED_EMAIL]" in output
    assert "[REDACTED_PHONE]" in output
    assert "[REDACTED_CARD]" in output
    assert "[REDACTED_ID]" in output
    assert "[REDACTED_IP]" in output
    assert "[REDACTED_URL]" in output
    assert "[REDACTED_ADDRESS]" in output
    assert "[REDACTED_ACCOUNT]" in output

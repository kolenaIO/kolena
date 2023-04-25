from kolena._utils.dataframes.validators import _is_locator_cell_valid


def test_validate_locator() -> None:
    valid_locators = [
        "s3://bucket-name/path/to/image.jpg",
        "gs://bucket/path/to/image.png",
        "s3://bucket/image with spaces.jpg",  # spaces should be allowed
        "s3://bucket/UPPERCASE.JPG",  # uppercase
        "gs://bucket/lower.jpG",
        "http://bucket/lower.jpG",
        "https://bucket/lower.jpG",
    ]
    invalid_locators = [
        "garbage",
        "closer://but/still/garbage.jpg",
        "s3://image.jpg",  # missing bucket name
        "s3://bucket/image.txt",  # non-image extension
        "s3:/bucket/image.jpg",  # malformed
    ]

    for locator in valid_locators:
        assert _is_locator_cell_valid(locator)

    for locator in invalid_locators:
        assert not _is_locator_cell_valid(locator)

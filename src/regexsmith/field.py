"""Hand-picked real-world formats.

They are never sampled for training, and they were written without looking at the
generator's output, so they test whether the model learned regexes or learned the
generator. Examples for them are still sampled from the regex.
"""

FIELD_TARGETS: tuple[tuple[str, str], ...] = (
    ("iso date", r"\d{4}-\d{2}-\d{2}"),
    ("us date", r"\d{2}/\d{2}/\d{4}"),
    ("clock time", r"\d{2}:\d{2}(:\d{2})?"),
    ("ipv4", r"\d{1,3}(\.\d{1,3}){3}"),
    ("mac address", r"([0-9a-f]{2}:){5}[0-9a-f]{2}"),
    ("uuid", r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
    ("hex colour", r"#[0-9a-f]{6}"),
    ("short git sha", r"[0-9a-f]{8}"),
    ("hex literal", r"0x[0-9a-f]+"),
    ("semver", r"\d+\.\d+\.\d+"),
    ("semver with tag", r"\d+\.\d+\.\d+(-[a-z0-9]+)?"),
    ("version tag", r"v\d+(\.\d+)?"),
    ("us zip", r"\d{5}(-\d{4})?"),
    ("us phone", r"\d{3}-\d{3}-\d{4}"),
    ("number plate", r"[A-Z]{3}-\d{4}"),
    ("money", r"\$\d+\.\d{2}"),
    ("percentage", r"\d{1,3}%"),
    ("signed float", r"-?\d+\.\d+"),
    ("csv of integers", r"\d+(,\d+)*"),
    ("slug", r"[a-z0-9]+(-[a-z0-9]+)*"),
    ("snake case", r"[a-z]+(_[a-z]+)*"),
    ("camel case", r"[a-z]+([A-Z][a-z]+)*"),
    ("pascal case", r"([A-Z][a-z]+)+"),
    ("constant case", r"[A-Z]+(_[A-Z]+)*"),
    ("hashtag", r"#\w+"),
    ("at mention", r"@[a-z0-9_]+"),
    ("env assignment", r"[A-Z_]+=.*"),
    ("key value", r"[a-z]+=[a-z0-9]+"),
    ("unix path", r"(/[a-z]+)+"),
    ("file name", r"[a-z]+\.(py|md|txt)"),
    ("http url", r"https?://[a-z]+\.(com|org)"),
    ("email", r"[a-z]+@[a-z]+\.(com|org|net)"),
    ("python dunder", r"__[a-z]+__"),
    ("ticket id", r"[A-Z]{2,4}-\d+"),
    ("bool word", r"true|false"),
    ("log level", r"INFO|WARN|ERROR"),
)

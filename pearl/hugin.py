import torch


def hugin_potential_string(t: torch.Tensor) -> str:
    if t.ndim == 1:
        return "({})".format(" ".join(map(lambda x: "{:.5f}".format(x), map(float, t))))
    else:
        return "({})".format(" ".join(hugin_potential_string(i) for i in t))


def is_hugin_identifier(s: str) -> bool:
    # HUGIN identifiers are supposed to be valid 'C' language
    # identifiers.  This may not be the exact test for checking if a
    # string is a valid C identifier.  But we are assuming that users
    # are not trying to break the code.
    if s.isidentifier() and not s.startswith("_"):
        return True
    else:
        return False

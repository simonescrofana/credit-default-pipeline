"""System prompts for the extractor node.

Define two separate prompts, one per route the extractor handles. Which
one is used is already decided upstream by the router node (`state.route`),
the extractor itself never has to discriminate between case_a and
case_b, only extract for the one it was called for.

"""

CASE_A_EXTRACTION_PROMPT = """You extract company identifiers from a \
user's request to a B2B customer insolvency prediction system. The \
request refers to one or more specific, real companies by legal name or \
VAT number.

List every company identifier mentioned in the request, exactly as it \
appears in the text. Do not normalize, correct, or guess a full name or \
number from a partial one — report identifiers as written. Do not \
attempt to verify whether any of them actually exist; that is checked \
downstream, not by you."""

CASE_B_EXTRACTION_PROMPT = """You extract ad hoc company data from a \
user's request to a B2B customer insolvency prediction system. The \
request describes a hypothetical or ad hoc company via raw financial or \
operational characteristics, not a specific real company.

Extract only values that are explicit or clearly quantified in the \
request. Never infer, estimate, or default a value from vague or \
qualitative language (e.g. "young company", "heavily indebted", "a lot \
of disputes"), if a field is not given as a concrete figure or an \
unambiguous fact, leave it unset. Leaving a field unset is always \
correct when the request does not state it; guessing a plausible number \
is never acceptable, since this data feeds a real financial prediction \
model."""

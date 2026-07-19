{% macro generate_company_key(company_id_col, valid_from_col) %}
{#
    Single source of truth for the company_key surrogate key.

    WHY THIS IS A MACRO AND NOT INLINE SQL: company_key is generated in two
    places int_companies_scd_resolved (where it becomes the FK written
    into fct_company_credit_profile) and dim_companies (where it's the PK).
    Both call sites MUST hash the exact same columns, in the exact same
    order, for the join between fact and dimension to resolve correctly
    a hash mismatch (even from something as small as reordering the
    columns, or renaming an alias) breaks that join *silently*: every FK in
    the fact table would point to a company_key that doesn't exist in the
    dimension, with no error, just rows that never match on select.

    Always call this macro from both places instead of writing
    generate_surrogate_key(...) by hand in more than one model.
#}
    {{ dbt_utils.generate_surrogate_key([company_id_col, valid_from_col]) }}
{% endmacro %}

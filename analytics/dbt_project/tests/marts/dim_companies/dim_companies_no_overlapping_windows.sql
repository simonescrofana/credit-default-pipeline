select
    a.company_id,
    a.dbt_valid_from as a_from,
    a.dbt_valid_to as a_to,
    b.dbt_valid_from as b_from,
    b.dbt_valid_to as b_to
from {{ ref('dim_companies') }} a
inner join {{ ref('dim_companies') }} b
    on a.company_id = b.company_id
    and a.company_key != b.company_key
    and a.dbt_valid_from < coalesce(b.dbt_valid_to, 'infinity'::timestamp)
    and coalesce(a.dbt_valid_to, 'infinity'::timestamp) > b.dbt_valid_from

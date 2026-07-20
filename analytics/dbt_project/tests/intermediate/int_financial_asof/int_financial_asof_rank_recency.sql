select
    f_asof.company_id,
    f_asof.snapshot_date,
    f_asof.fiscal_year as chosen_fiscal_year,
    stg.fiscal_year as newer_available_fiscal_year
from {{ ref('int_financial_asof') }} f_asof
inner join {{ ref('stg_financial_statements') }} stg
    on f_asof.company_id = stg.company_id
    and stg.fiscal_year > f_asof.fiscal_year
    and make_date(stg.fiscal_year + 1, 6, 30) <= f_asof.snapshot_date

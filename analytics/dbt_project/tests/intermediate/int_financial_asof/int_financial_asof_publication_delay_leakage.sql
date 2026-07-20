select
    company_id,
    snapshot_date,
    fiscal_year
from {{ ref('int_financial_asof') }}
where snapshot_date < make_date(fiscal_year + 1, 6, 30)

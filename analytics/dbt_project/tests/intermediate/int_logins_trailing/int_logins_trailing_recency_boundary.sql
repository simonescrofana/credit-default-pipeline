select
    company_id,
    snapshot_date,
    days_since_last_login
from {{ ref('int_logins_trailing') }}
where days_since_last_login >= 90

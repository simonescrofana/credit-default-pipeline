select
    company_id,
    snapshot_date,
    login_velocity
from {{ ref('int_logins_trailing') }}
where login_velocity > 3.0001

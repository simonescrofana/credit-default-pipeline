select
    company_id,
    snapshot_date,
    days_since_last_login,
    login_velocity
from {{ ref('int_logins_trailing') }}
where (days_since_last_login is null and login_velocity is not null)
   or (days_since_last_login is not null and login_velocity is null)

select company_id, count(*) as current_versions
from {{ ref('dim_companies') }}
where dbt_valid_to is null
group by company_id
having count(*) > 1

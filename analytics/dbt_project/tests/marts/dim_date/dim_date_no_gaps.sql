with numbered as (
    select
        date_day,
        row_number() over (order by date_day) as rn
    from {{ ref('dim_date') }}
)

select date_day
from numbered
where date_day != (select min(date_day) from numbered) + (rn - 1) * interval '1 day'

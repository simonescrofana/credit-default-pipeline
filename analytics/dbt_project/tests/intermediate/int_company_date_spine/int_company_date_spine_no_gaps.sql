with numbered as (
    select
        company_id,
        snapshot_date,
        row_number() over (partition by company_id order by snapshot_date) as rn,
        (extract(year from snapshot_date) * 12 + extract(month from snapshot_date))::int
            as month_index
    from {{ ref('int_company_date_spine') }}
),
flagged as (
    select
        company_id,
        snapshot_date,
        month_index - rn as gap_marker,
        min(month_index - rn) over (partition by company_id) as expected_marker
    from numbered
)
select company_id, snapshot_date
from flagged
where gap_marker != expected_marker
-- the logic is that the month_index - rn keeps constant if no month is missing (for each company)
-- therefore if a month is missing the value of month_index - rn is not constant anymore which
-- means also "not corresponding to its minimum", since a missing month means an increase of the quantity
-- month_index - rn, due to the fact that rn grows always by 1 but skipping month means increase by 2,
-- resulting in adding 1 to the value of month_index - rn (for each skipped month) which is bigger than
-- min(month_index - rn)

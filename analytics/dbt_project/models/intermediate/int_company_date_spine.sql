{{ config(materialized='table') }}

with companies as (
    select distinct
        company_id,
        foundation_date,
        dbt_valid_to
    from {{ ref('stg_companies') }}
),

month_spine as (
    {{ dbt_utils.date_spine(
        datepart="month",
        start_date="(select date_trunc('month', min(foundation_date)) from " ~ ref('stg_companies') ~ ")",
        end_date="date_trunc('month', current_date) + interval '1 month'"
    ) }}
),

spine as (
    select
        c.company_id,
        m.date_month::date as snapshot_date
    from companies as c
    inner join month_spine as m
        on m.date_month >= date_trunc('month', c.foundation_date)
        and (c.dbt_valid_to is null or m.date_month < c.dbt_valid_to)
        and m.date_month <= date_trunc('month', current_date)
)

select * from spine

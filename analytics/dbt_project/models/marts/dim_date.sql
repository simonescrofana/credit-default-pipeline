with date_spine as (
    {{ dbt_utils.date_spine(
        datepart="day",
        start_date="(select min(foundation_date) from " ~ ref('stg_companies') ~ ")",
        end_date="current_date + interval '1 day'"
    ) }}
),

final as (
    select
        date_day::date as date_day,
        extract(month from date_day)::int as month_number,
        extract(quarter from date_day)::int as quarter,
        extract(year from date_day)::int as year,
        -- Postgres extract(dow ...) is 0=Sunday..6=Saturday
        extract(dow from date_day)::int as day_of_week,
        trim(to_char(date_day, 'Day')) as day_name,
        extract(dow from date_day) in (0, 6) as is_weekend
    from date_spine
)

select * from final

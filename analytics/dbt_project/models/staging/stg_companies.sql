with source as (
    select * from {{ source('db_source', 'companies') }}
),

transformed as (
    select
        id::bigint as company_id,
        vat_number as vat_number,
        trim(legal_name) as legal_name,
        upper(trim(legal_form)) as legal_form,
        lower(trim(industry_sector)) as industry_sector,
        foundation_date::date as foundation_date,
        (current_date - foundation_date::date)::int as company_age_days,
        coalesce(trim(registered_office_region), 'unknown') as registered_office_region,
        coalesce(is_active, true) as is_active

    from source
)

select * from transformed

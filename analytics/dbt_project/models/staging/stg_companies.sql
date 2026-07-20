with source as (
    select * from {{ ref('companies_snapshot') }}
),

transformed as (
    select
        id::bigint as company_id,
        vat_number as vat_number,
        trim(legal_name) as legal_name,
        upper(trim(legal_form)) as legal_form,
        lower(trim(industry_sector)) as industry_sector,
        foundation_date::date as foundation_date,
        coalesce(trim(registered_office_region), 'unknown') as registered_office_region,
        coalesce(is_active, true) as is_active,
        dbt_valid_from,
        dbt_valid_to

    from source
)

select * from transformed

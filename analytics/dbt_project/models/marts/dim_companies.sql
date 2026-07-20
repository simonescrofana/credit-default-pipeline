with companies as (
    select * from {{ ref('stg_companies') }}
),

final as (
    select
        {{ generate_company_key('company_id', 'dbt_valid_from') }} as company_key,
        company_id,
        vat_number,
        legal_name,
        legal_form,
        foundation_date,
        is_active,
        registered_office_region,
        industry_sector,
        dbt_valid_from,
        dbt_valid_to
    from companies
)

select * from final

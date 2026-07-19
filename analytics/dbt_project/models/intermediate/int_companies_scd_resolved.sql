with spine as (
    select * from {{ ref('int_company_date_spine') }}
),

companies as (
    select * from {{ ref('stg_companies') }}
),

resolved as (
    select
        spine.company_id,
        spine.snapshot_date,
        {{ generate_company_key('companies.company_id', 'companies.dbt_valid_from') }}
            as company_key,
        companies.vat_number,
        companies.legal_name,
        companies.legal_form,
        companies.foundation_date,
        companies.is_active,
        companies.registered_office_region,
        companies.industry_sector,
        (spine.snapshot_date - companies.foundation_date) as company_age_days
    from spine
    inner join companies
        on spine.company_id = companies.company_id
        -- this means "all snapshots"
        and spine.snapshot_date >= companies.dbt_valid_from
        and (
            spine.snapshot_date < companies.dbt_valid_to
            or companies.dbt_valid_to is null
        )
)
select * from resolved

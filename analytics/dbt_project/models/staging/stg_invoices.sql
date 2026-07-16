with source as (
    select * from {{ source('db_source', 'invoices') }}
),

transformed as (
    select
        id::bigint as invoice_id,
        contract_id::bigint as contract_id,
        lower(trim(commodity_type)) as commodity_type,
        trim(invoice_number) as invoice_number,
        lower(trim(invoice_status)) as invoice_status,
        electricity_consumption_kwh::decimal(10,2) as electricity_consumption_kwh,
        gas_consumption_scm::decimal(10,2) as gas_consumption_scm,
        amount_excluding_tax::decimal(15,2) as amount_excluding_tax,
        tax_amount::decimal(15,2) as tax_amount,
        total_amount::decimal(15,2) as total_amount,
        issue_date::date as issue_date,
        due_date::date as due_date,
        (due_date::date - issue_date::date)::int as payment_term_days

    from source
)

select * from transformed

with source as (
    select * from {{ source('db_source', 'payments') }}
),

transformed as (
    select
        id::bigint as payment_id,
        invoice_id::bigint as invoice_id,
        trim(transaction_reference) as transaction_reference,
        lower(trim(payment_method)) as payment_method,
        lower(trim(payment_status)) as payment_status,
        amount_paid::decimal(15,2) as amount_paid,
        payment_date::date as payment_date

    from source
)

select * from transformed

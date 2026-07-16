with source as (
    select * from {{ source('db_source', 'crm_support_tickets') }}
),

transformed as (
    select
        id::bigint as ticket_id,
        company_id::bigint as company_id,
        lower(trim(ticket_category)) as ticket_category,
        satisfaction_score::int as satisfaction_score,
        created_at::timestamptz as created_at,
        resolved_at::timestamptz as resolved_at,
        case 
            when resolved_at is not null then 
                (extract(epoch from (resolved_at::timestamptz - created_at::timestamptz)) / 3600.0)::decimal(10,2)
            else null
        end as resolution_time_hours,
        case 
            when resolved_at is not null then 
                (extract(epoch from (resolved_at::timestamptz - created_at::timestamptz)) / 86400.0)::decimal(10,2)
            else null
        end as resolution_time_days

    from source
)

select * from transformed

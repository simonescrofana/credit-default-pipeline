with source as (
    select * from {{ source('db_source', 'user_web_logins') }}
),

transformed as (
    select
        id::bigint as login_id,
        company_id::bigint as company_id,
        user_id::bigint as user_id,
        login_timestamp::timestamptz as login_timestamp,
        ip_address::varchar(45) as ip_address, 
        lower(trim(device_type)) as device_type

    from source
)

select * from transformed
